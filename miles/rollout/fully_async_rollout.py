"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import heapq
import logging
import math
import time
from dataclasses import replace

from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    Group,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.function_registry import load_function
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else self._forget_attempts
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        # The loop the worker runs on; the producer calls into the data source from its thread.
        self._producer_loop: asyncio.AbstractEventLoop | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        self._max_prompt_attempts = getattr(input.args, "async_max_prompt_attempts", None)
        self._retry_backoff_s = getattr(input.args, "async_retry_backoff_s", 0.0)
        self._no_progress_timeout_s = getattr(input.args, "async_no_progress_timeout_s", None)
        if self._max_prompt_attempts is not None and self._max_prompt_attempts <= 0:
            raise ValueError("async_max_prompt_attempts must be positive")
        if not math.isfinite(self._retry_backoff_s) or self._retry_backoff_s < 0:
            raise ValueError("async_retry_backoff_s must be finite and nonnegative")
        if self._no_progress_timeout_s is not None and (
            not math.isfinite(self._no_progress_timeout_s) or self._no_progress_timeout_s <= 0
        ):
            raise ValueError("async_no_progress_timeout_s must be positive and finite")
        self._retry_counts: dict[int, int] = {}
        self._retry_queue: list[tuple[float, int, list[Sample]]] = []
        self._retry_sequence = 0
        self._retries_exhausted = 0

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or DefaultDataBuffer
            self._output = buffer_cls(
                DataBufferConstructorInput(
                    args=self.args,
                    unused_handler_fn=self._handle_unused,
                    discard_handler_fn=self._forget_attempts,
                )
            )
            self._worker = asyncio.create_task(self._worker_loop())
            self._producer_loop = asyncio.get_running_loop()
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> tuple[asyncio.Task, list[Sample]]:
        if self._retry_queue and self._retry_queue[0][0] <= time.monotonic():
            _, _, prompt_group = heapq.heappop(self._retry_queue)
            samples = [prompt_group]
        else:
            samples = self.data_source.get_samples(1)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        if self._max_prompt_attempts is not None or self._retry_backoff_s:
            group_id = self._group_id(prompt_group)
            self._retry_counts[group_id] = self._retry_counts.get(group_id, 0) + 1
        return asyncio.create_task(self._generate_group(prompt_group)), prompt_group

    @staticmethod
    def _group_id(prompt_group: list[Sample]) -> int:
        group_id = prompt_group[0].group_index
        if type(group_id) is not int or group_id < 0:
            raise ValueError("bounded retries require a nonnegative prompt group_index")
        return group_id

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self) -> None:
        active: dict[asyncio.Task, list[Sample]] = {}
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                task, prompt_group = self._submit_one_group()
                active[task] = prompt_group
            done, _ = await self._scheduler.wait_for_progress(set(active))
            for task in done:
                prompt_group = active.pop(task)
                entry = self._collect_group_result(task, prompt_group)
                if self._source_takes(entry.group):
                    self._forget_attempts(prompt_group)
                    continue
                await self._output.put(entry)

    def _source_takes(self, group: Group) -> bool:
        """Show the data source each finished group; it may keep one out of training, e.g. an eval probe."""
        observe = getattr(self.data_source, "on_group_finished", None)
        return observe is not None and bool(observe(group))

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
        )

    # -------------------------- consumer --------------------------

    async def _next_group(self, current_version: int | None, *, deadline: float | None = None) -> DataBufferInput:
        queue_get = asyncio.create_task(self._output.get(current_version=current_version))
        try:
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(
                        "fully-async rollout made no accepted-group progress; "
                        f"retry_exhausted={self._retries_exhausted}, queued_retries={len(self._retry_queue)}"
                    )
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=min(NO_PROGRESS_WARN_SECS, remaining) if remaining is not None else NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True

        while len(data) < target_data_size:
            deadline = (
                time.monotonic() + self._no_progress_timeout_s if self._no_progress_timeout_s is not None else None
            )
            entry = await self._next_group(input.weight_version, deadline=deadline)
            assert len(entry.group) == args.n_samples_per_prompt
            self._forget_attempts(entry.prompt_group)

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        metrics = self._output.get_metrics()
        metrics["rollout/fully_async/retry_exhausted"] = self._retries_exhausted
        metrics["rollout/fully_async/queued_retries"] = len(self._retry_queue)
        if (source_metrics := getattr(self.data_source, "rollout_metrics", None)) is not None:
            metrics.update(source_metrics())
        return RolloutFnTrainOutput(samples=data, metrics=metrics)

    def _recycle(self, prompt_group: list[Sample]) -> None:
        if self._max_prompt_attempts is not None or self._retry_backoff_s:
            group_id = self._group_id(prompt_group)
            attempts = self._retry_counts.get(group_id, 0)
            if self._max_prompt_attempts is not None and attempts >= self._max_prompt_attempts:
                self._forget_attempts(prompt_group)
                self._retries_exhausted += 1
                logger.warning("Quarantined prompt group after %d generation attempts", attempts)
                return
        for sample in prompt_group:
            sample.reset_for_retry()
        if self._retry_backoff_s:
            delay = min(self._retry_backoff_s * 2 ** max(0, attempts - 1), 30.0)
            self._retry_sequence += 1
            heapq.heappush(
                self._retry_queue,
                (time.monotonic() + delay, self._retry_sequence, prompt_group),
            )
        else:
            self.data_source.add_samples([prompt_group])

    def _forget_attempts(self, prompt_group: list[Sample]) -> None:
        if self._max_prompt_attempts is not None or self._retry_backoff_s:
            self._retry_counts.pop(self._group_id(prompt_group), None)

    # -------------------------- checkpointing --------------------------

    def save_with_data_source(self, rollout_id: int) -> None:
        """Save the data source on the producer's loop.

        The producer calls the source's get_samples, add_samples and on_group_finished from that loop's
        thread while the trainer saves. A save from the rollout actor's thread could read the source
        mid-update: a dict that changes size during iteration, or a cursor out of step with the groups
        it has issued. Run on the loop, the save sits between two producer steps.
        """
        loop = self._producer_loop
        if loop is None or not loop.is_running() or _running_loop() is loop:
            self.data_source.save(rollout_id)
            return

        async def save() -> None:
            self.data_source.save(rollout_id)

        asyncio.run_coroutine_threadsafe(save(), loop).result()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None
