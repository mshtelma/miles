"""
Generic agentic generate function for agent-environment RL training.

The agent logic is fully encapsulated in a user-provided async function
(--custom-agent-function-path). This generate function only handles:
  1. TITO session tracing (OpenAIEndpointTracer)
  2. Collecting the worker-assembled training samples (the session server
     converts records to samples, truncates and merges in the owning worker)
  3. Driver-side metadata application (agent_metadata, session_metadata)

Agent function contract:
  async def my_agent(
      base_url: str,
      prompt: ...,
      request_kwargs: dict,
      metadata: dict,       # sample.metadata — env-specific fields
      **kwargs,
  ) -> dict | None:
      ...

  Returning None means no extra metadata to attach.
  Returning a dict merges it into every sample's metadata, so downstream
  reward models (--custom-rm-path) can read whatever the agent left there.
"""

import argparse
import asyncio
import logging
import math
import os
import time
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import httpx
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.agent_errors import EnvironmentFailure
from miles.rollout.generate_utils.openai_endpoint_utils import SESSION_COLLECT_TIMEOUT, OpenAIEndpointTracer
from miles.rollout.session.samples.provenance import untrusted_agent_metadata
from miles.rollout.session.v2.metrics import SESSION_ROLLOUT_METRICS_KEY
from miles.utils.function_registry import load_function
from miles.utils.http_utils import HTTPResponseError
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

# The session server's assembly plus decoding the returned samples here.
_COLLECT_TIMEOUT = SESSION_COLLECT_TIMEOUT + 300


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    assert not input.args.partial_rollout, "Partial rollout is not supported"
    assert getattr(input.args, "session_server_addrs", None), (
        "agentic_tool_call.generate requires session_server_addrs. "
        "Pass --use-session-server to start the session server."
    )
    use_v2 = getattr(input.args, "use_session_server", None) == "v2"
    collect_spec_metrics = use_v2 and input.args.sglang_speculative_algorithm is not None
    custom_agent_function: Callable = load_function(input.args.custom_agent_function_path)
    assert (
        custom_agent_function is not None
    ), f"Custom agent function {input.args.custom_agent_function_path} not found"
    tracer = await OpenAIEndpointTracer.create(
        input.args, evaluation=input.evaluation, sampling_params=input.sampling_params
    )

    max_seq_len = getattr(input.args, "max_seq_len", None)

    metadata = input.sample.metadata
    if max_seq_len is not None:
        metadata = {**metadata, "max_seq_len": max_seq_len}
    if tracer.session_server_instance_id:
        metadata = {**metadata, "session_server_instance_id": tracer.session_server_instance_id}

    log_prefix = f"[session={tracer.session_id}]"

    # From the tracer, not args: with multiple instances the owning ip:port is per-session.
    metadata = {
        **metadata,
        "session_server_id": tracer.session_server_id,
        "session_id": tracer.session_id,
        "attempt_id": os.environ.get("MILES_ATTEMPT_ID", getattr(input.args, "run_id", "local")),
    }

    agent_metadata = None
    collect_failed = False
    agent_failed = False
    agent_exception = None
    cancelled = False
    non_generation_time = None
    strict_errors = bool(getattr(input.args, "agentic_abort_on_agent_error", False))
    t_start = time.monotonic()
    try:
        logger.debug(f"{log_prefix} Starting agent function call")
        agent_metadata = await custom_agent_function(
            base_url=tracer.base_url,
            prompt=input.sample.prompt,
            request_kwargs=build_chat_request_kwargs(input.sampling_params),
            metadata=metadata,
        )
        if agent_metadata is not None and not isinstance(agent_metadata, dict):
            agent_metadata = None
            raise TypeError("agent function must return a metadata object or None")
        non_generation_time = _agent_tool_time(agent_metadata)
        logger.debug(f"{log_prefix} Agent function returned in {time.monotonic()-t_start:.1f}s")
    except asyncio.CancelledError:
        cancelled = True
        raise
    except EnvironmentFailure as e:
        agent_failed = True
        logger.warning(f"{log_prefix} Agent environment failed: {e}", exc_info=True)
    except Exception as e:
        if strict_errors:
            agent_exception = e
            logger.error(f"{log_prefix} Agent implementation failed: {e}", exc_info=True)
        else:
            logger.warning(f"{log_prefix} Agent function failed: {e}", exc_info=True)

    finally:
        if cancelled:
            # A cancelled caller has no training sample to collect. Delete the
            # session promptly so its recorded turns and active backend work do
            # not remain attached to an abandoned request.
            try:
                await tracer.delete_session(timeout=10)
            except Exception as e:
                logger.warning(f"{log_prefix} Failed deleting cancelled session: {e!r}", exc_info=True)
        else:
            # Both session versions attest the retained terminal response
            # against the agent's scored response ID during collection.
            collect_kwargs = {"max_seq_len": max_seq_len, "agent_metadata": agent_metadata}
            try:
                result = await asyncio.wait_for(
                    tracer.collect_samples(input.sample, **collect_kwargs), timeout=_COLLECT_TIMEOUT
                )
            except (TimeoutError, httpx.TransportError) as e:
                collect_failed = True
                logger.warning(f"{log_prefix} Failed collecting samples: {e!r}", exc_info=True)
            except HTTPResponseError as e:
                if not strict_errors or (e.status_code != 429 and e.status_code < 500):
                    if agent_exception is None:
                        raise
                    logger.warning(f"{log_prefix} Collection also failed after an agent error: {e!r}")
                else:
                    collect_failed = True
                    logger.warning(f"{log_prefix} Session collector unavailable: {e!r}", exc_info=True)
            except Exception as e:
                if agent_exception is None:
                    raise
                logger.warning(f"{log_prefix} Collection also failed after an agent error: {e!r}", exc_info=True)
            else:
                logger.debug(
                    f"{log_prefix} collect_samples done: {len(result.samples)} samples, "
                    f"total_time={time.monotonic()-t_start:.1f}s"
                )

    if agent_exception is not None:
        raise agent_exception

    if collect_failed or (agent_failed and strict_errors):
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        if collect_spec_metrics:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
        return GenerateFnOutput(samples=[sample] if use_v2 else sample)

    if not result.samples:
        if result.empty_reason == "all_truncated":
            logger.warning("All samples truncated (prompt already exceeds max_seq_len)")
        else:
            logger.warning("No model calls recorded for sample")
        sample = deepcopy(input.sample)
        sample.status = Sample.Status.ABORTED
        if collect_spec_metrics:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
        return GenerateFnOutput(samples=[sample] if use_v2 else sample)

    session_rollout_metrics = None
    if collect_spec_metrics:
        session_rollout_metrics = result.session_metadata[SESSION_ROLLOUT_METRICS_KEY]
        if session_rollout_metrics["session_id"] != tracer.session_id:
            raise ValueError(
                "session_rollout_metrics.session_id does not match the collected session: "
                f"{session_rollout_metrics['session_id']!r} != {tracer.session_id!r}"
            )
        if session_rollout_metrics["metrics"] is None:
            raise ValueError("a successful session collect must carry metrics")

    samples = result.samples
    if collect_spec_metrics:
        for sample in samples:
            sample.metadata.pop(SESSION_ROLLOUT_METRICS_KEY, None)
            sample.metadata[SESSION_ROLLOUT_METRICS_KEY] = session_rollout_metrics
    if use_v2 and len(samples) > 1:
        # FIXME: handle sample index issues.
        rollout_id = input.sample.rollout_id if input.sample.rollout_id is not None else input.sample.index
        assert rollout_id is not None, "v2 agentic samples require input Sample.rollout_id or Sample.index"
        for sample in samples:
            sample.rollout_id = rollout_id
    if not use_v2:
        # v1: the agent's metadata is applied driver-side. Under v2 it traveled
        # through collect_samples and came back applied by the server-side
        # merge (per-sample metadata/reward on the wire) — no overlay here.
        for s in samples:
            s.metadata.update(untrusted_agent_metadata(agent_metadata))

    # If the agent function reports wall-clock time spent outside policy generation
    # (env/tool steps), surface it on Sample.non_generation_time so throughput
    # accounting subtracts it.
    if non_generation_time is not None:
        for s in samples:
            s.non_generation_time = non_generation_time

    if use_v2:
        return GenerateFnOutput(samples=samples)

    (sample,) = samples
    sample.metadata.update(result.session_metadata)
    return GenerateFnOutput(samples=sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--custom-agent-function-path", type=str)
    parser.add_argument(
        "--agentic-abort-on-agent-error",
        action="store_true",
        help="Abort environment failures and fail fast on agent implementation errors after session cleanup.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        dest="max_seq_len",
        help="Max sequence length in tokens (prompt + completion, including env responses) "
        "per session. Truncation happens inside the session server during sample assembly; "
        "also forwarded to the Harbor agent server (as max_seq_len) to abort the trial early.",
    )


generate.add_arguments = _add_arguments


def _agent_tool_time(agent_metadata: dict | None) -> float | None:
    if agent_metadata is None:
        return None
    metrics = agent_metadata.get("agent_metrics")
    if metrics is None:
        return None
    if not isinstance(metrics, dict):
        raise TypeError("agent_metrics must be an object")
    value = metrics.get("total_tool_time")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("agent total_tool_time must be finite and nonnegative")
    return float(value)


# Process keys to match ChatCompletionRequest input
def build_chat_request_kwargs(sampling_params: dict[str, Any]) -> dict[str, Any]:
    request_kwargs = dict(sampling_params)
    key_map = {
        "max_new_tokens": "max_tokens",
        "min_new_tokens": "min_tokens",
        "sampling_seed": "seed",
    }
    for src, dst in key_map.items():
        if src in request_kwargs:
            if dst not in request_kwargs:
                request_kwargs[dst] = request_kwargs[src]
            request_kwargs.pop(src, None)

    reserved_keys = {"model", "messages"}
    allowed_keys = set(ChatCompletionRequest.model_fields) - reserved_keys
    return {key: value for key, value in request_kwargs.items() if key in allowed_keys and value is not None}
