"""Evaluate an HF checkpoint on OfficeQA with one standalone SGLang server on one AIR node.

The checkpoint is the base model or the HF export inside a training save that has been copied to the Volume
in full: Megatron's tracker there has reached the save, and Miles marked its export complete.
"""

import argparse
import asyncio
import collections
import faulthandler
import json
import logging
import os
import shutil
import time
from pathlib import Path

from examples.air_officeqa import settings
from examples.air_officeqa.eval import driver, proxy, server
from examples.air_officeqa.models import MODELS, Model
from tools.air.copy_tree import copy_tree

logger = logging.getLogger(__name__)

# Written next to the tracker by tools/air/publisher.py, which this module does not import: it needs Ray.
MLFLOW_RUNS = "mlflow_runs.json"
# A hung eval otherwise runs on silently until AIR's idle-GPU kill, with no sign of where it stopped.
STALL_DUMP_S = 600


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", choices=sorted(MODELS), default="GLM-5.3-Flash")
    parser.add_argument("--local-root", type=Path, default=Path("/tmp/miles-air"))
    parser.add_argument("--hf-source", default="base", help="base, or a save's directory .../checkpoints/iter_<step>")
    parser.add_argument("--eval-set", action="append", metavar="NAME=PATH")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=480)
    parser.add_argument("--max-seq-len", type=int, default=150_000)
    parser.add_argument("--max-lost-fraction", type=float, default=0.05)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    args = parser.parse_args(argv)
    args.attempt_id = os.environ["MLFLOW_RUN_ID"]
    args.out_dir = args.volume_root / "runs" / args.run_id / args.attempt_id
    args.eval_sets = _eval_sets(args.eval_set or ["heldout=data/officeqa/eval_heldout.jsonl"], args.local_root)
    return args


def model_source(volume_root: Path, hf_source: str, *, model_name: str) -> tuple[Path, dict]:
    """The HF directory to serve, and what the outputs record about it."""
    if hf_source == "base":
        return volume_root / "models" / model_name, {"kind": "base"}
    save = Path(hf_source)
    step = int(save.name.removeprefix("iter_"))
    copied = int((save.parent / "latest_checkpointed_iteration.txt").read_text())
    if copied < step or not (save / "hf/.complete").is_file():
        raise ValueError(f"{save} is not a fully copied save with a complete HF export")
    runs_file = save.parent / MLFLOW_RUNS
    training_run = json.loads(runs_file.read_text()).get(str(step)) if runs_file.is_file() else None
    return save / "hf", {
        "kind": "checkpoint",
        "path": str(save),
        "rollout_id": step,
        **({"mlflow_run_id": training_run} if training_run else {}),
    }


def _eval_sets(values: list[str], input_root: Path) -> dict[str, Path]:
    sets = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path or name in sets:
            raise ValueError(f"--eval-set needs a unique NAME=PATH: {value!r}")
        sets[name] = Path(path) if Path(path).is_absolute() else input_root / path
    return sets


def _tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)


def _token_counter(tokenizer, profile: Model):
    """Count a prompt's tokens as the server renders it: the training chat template, its kwargs, and tools
    normalized as in training."""
    from miles.utils.chat_template_utils.template import apply_chat_template

    template, template_kwargs = server.chat_template(profile)
    if template is not None:
        template_kwargs = {**template_kwargs, "chat_template": Path(template).read_text()}

    def count(messages, tools):
        return len(
            apply_chat_template(
                messages,
                tokenizer=tokenizer,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **template_kwargs,
            )
        )

    return count


def _stop_texts(model_path: Path, tokenizer) -> tuple[str, ...]:
    """The checkpoint's stop tokens as text (generation_config.json ``eos_token_id``)."""
    eos = json.loads((model_path / "generation_config.json").read_text())["eos_token_id"]
    return tuple(tokenizer.convert_ids_to_tokens(eos if isinstance(eos, list) else [eos]))


class _StallWatch:
    """Dump the stacks when the eval stops making progress.

    faulthandler's timer, re-armed on every tick, dumps every thread's stack if the event loop itself stops,
    even while a thread holds the GIL. If the loop runs but no model turn finishes, ``_dump_stacks`` runs once.
    """

    def __init__(self, sglang_log: Path, *, stall_s: float = STALL_DUMP_S, clock=time.monotonic):
        self._sglang_log = sglang_log
        self._stall_s = stall_s
        self._clock = clock
        self._turns, self._progressed_at, self._dumped = 0, clock(), False

    def tick(self, turns: int) -> None:
        faulthandler.dump_traceback_later(self._stall_s, repeat=True)
        now = self._clock()
        if turns != self._turns:
            self._turns, self._progressed_at, self._dumped = turns, now, False
        elif not self._dumped and now - self._progressed_at > self._stall_s:
            self._dumped = True
            _dump_stacks(f"no model turn finished for {now - self._progressed_at:.0f} s", self._sglang_log)


def _dump_stacks(reason: str, sglang_log: Path) -> None:
    """Every thread's stack, where the pending tasks wait, and the end of the SGLang log."""
    logger.error("%s; the stacks of all threads, the pending tasks and the end of the SGLang log follow", reason)
    faulthandler.dump_traceback(all_threads=True)
    waits = collections.Counter(_await_chain(task.get_coro()) for task in asyncio.all_tasks())
    for chain, count in waits.most_common():
        logger.error("%5d tasks wait at %s", count, chain)
    if sglang_log.exists():
        tail = sglang_log.read_text(errors="replace").splitlines()[-50:]
        logger.error("SGLang log, last %d lines:\n%s", len(tail), "\n".join(tail))


def _await_chain(coro) -> str:
    """The frames of a suspended coroutine and of the coroutines it awaits, outermost first."""
    frames = []
    while (frame := getattr(coro, "cr_frame", None)) is not None:
        frames.append(f"{Path(frame.f_code.co_filename).name}:{frame.f_lineno} {frame.f_code.co_name}")
        coro = coro.cr_await
    return " > ".join(frames) or repr(coro)


async def _evaluate(args, url: str, model_path: Path, process, *, sglang_log: Path):
    tokenizer = _tokenizer(model_path)
    chat = proxy.ChatProxy(
        url,
        count_prompt_tokens=_token_counter(tokenizer, MODELS[args.model_name]),
        max_seq_len=args.max_seq_len,
        stop_texts=_stop_texts(model_path, tokenizer),
    )
    runner, proxy_url = await proxy.serve(chat)
    task = None
    try:
        # The agent reads the environment set by main.
        from examples.air_officeqa.agent import run as agent_run

        sets = {name: driver.load_rows(path)[: args.limit_questions or None] for name, path in args.eval_sets.items()}
        task = asyncio.create_task(
            driver.run_sets(
                sets,
                agent_run=agent_run,
                proxy=chat,
                proxy_url=proxy_url,
                samples_per_prompt=args.samples_per_prompt,
                max_concurrent=args.max_concurrent,
                max_seq_len=args.max_seq_len,
                attempt_id=args.attempt_id,
            )
        )
        watch = _StallWatch(sglang_log)
        while not task.done():
            if process.poll() is not None:
                raise RuntimeError(f"SGLang exited with code {process.returncode} during the eval")
            watch.tick(len(chat.records))
            await asyncio.wait([task], timeout=5)
        return await task, chat.summary()
    finally:
        faulthandler.cancel_dump_traceback_later()
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()


def _log_mlflow(args, metrics, model, step, *, valid):
    """Log into the eval's own run and, for a valid eval of a save, into the training run at the save's
    step, where the held-out scores sit next to the training curves."""
    try:
        import mlflow
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
        with mlflow.start_run(run_id=args.attempt_id):
            mlflow.set_tag("mlflow.runName", args.run_id)
            mlflow.log_params({"hf_source": args.hf_source, "model": json.dumps(model, sort_keys=True)[:500]})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)
        if valid and "mlflow_run_id" in model:
            now = int(time.time() * 1000)
            scores = [
                Metric(key, float(value), now, step)
                for key, value in metrics.items()
                if not key.startswith("officeqa_eval/harness/")
            ]
            MlflowClient().log_batch(model["mlflow_run_id"], metrics=scores)
    except Exception:
        logger.exception("MLflow logging failed; eval results are on the Volume")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse(argv)
    started = time.monotonic()
    source, model = model_source(args.volume_root, args.hf_source, model_name=args.model_name)
    model_path = args.local_root / "model"
    copy_tree(source, model_path)
    copy_tree(args.volume_root / "data/officeqa", args.local_root / "data/officeqa")
    os.environ.update(settings.agent_env(str(args.local_root / "data")))
    timings = {"stage_s": time.monotonic() - started}
    url = f"http://127.0.0.1:{args.port}"
    log_path = args.local_root / "sglang.log"
    process = server.launch(
        server.server_argv(MODELS[args.model_name], model_path, host="127.0.0.1", port=args.port, seed=1234),
        log_path=log_path,
    )
    try:
        timings["ready_s"] = server.wait_ready(url, process, timeout_s=1800)
        started = time.monotonic()
        outcomes, harness = asyncio.run(_evaluate(args, url, model_path, process, sglang_log=log_path))
        timings["eval_s"] = time.monotonic() - started
    finally:
        server.stop(process)
        if log_path.exists():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log_path, args.out_dir / "sglang.log")
    lost = driver.lost_fraction(outcomes)
    valid = lost <= args.max_lost_fraction
    metrics = {
        key: value for name, outcome in outcomes.items() for key, value in driver.set_metrics(name, outcome).items()
    }
    metrics.update(
        {f"officeqa_eval/harness/{key}": value for key, value in {**harness, **timings, "lost_fraction": lost}.items()}
    )
    summary = {"valid": valid, "metrics": metrics, "model": model}
    driver.write_outputs(args.out_dir, model=model, summary=summary, outcomes=outcomes)
    _log_mlflow(args, metrics, model, model.get("rollout_id") or 0, valid=valid)
    print("OfficeQA eval: " + json.dumps({"valid": valid, "lost_fraction": lost, "model": model}), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
