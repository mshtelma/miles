"""Strict agent failure classification preserves the session collector contract."""

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from miles.rollout.generate_utils.agent_errors import EnvironmentFailure
from miles.utils.http_utils import HTTPResponseError
from miles.utils.types import Sample


class _Tracer:
    session_id = "session-1"
    session_server_id = "127.0.0.1:12345"
    session_server_instance_id = None
    base_url = "http://127.0.0.1:12345/sessions/session-1"

    def __init__(self, *, error=None, samples=None):
        self.error = error
        self.samples = samples if samples is not None else [Sample(status=Sample.Status.COMPLETED)]
        self.collected = []
        self.deleted = []

    async def collect_samples(self, input_sample, *, max_seq_len, agent_metadata=None):
        self.collected.append((input_sample, max_seq_len, agent_metadata))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(samples=self.samples, session_metadata={}, empty_reason=None)

    async def delete_session(self, *, timeout):
        self.deleted.append(timeout)


def _input(*, version: str, strict: bool = True):
    args = SimpleNamespace(
        partial_rollout=False,
        session_server_addrs=["127.0.0.1:12345"],
        use_session_server=version,
        sglang_speculative_algorithm=None,
        custom_agent_function_path="tests.fake_agent",
        max_seq_len=64,
        agentic_abort_on_agent_error=strict,
        run_id="run-1",
    )
    sample = Sample(index=3, group_index=1, prompt="raw prompt", metadata={"task_id": "task-1"})
    return SimpleNamespace(
        state=SimpleNamespace(args=args),
        args=args,
        sample=sample,
        sampling_params={},
        evaluation=False,
    )


def _patch(monkeypatch, agent, tracer):
    wrapper = importlib.import_module("miles.rollout.generate_hub.agentic_tool_call")

    async def create(args, *, evaluation=False, sampling_params=None):
        return tracer

    monkeypatch.setattr(wrapper.OpenAIEndpointTracer, "create", create)
    monkeypatch.setattr(wrapper, "load_function", lambda path: agent)
    return wrapper


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_environment_failure_aborts_both_session_versions(monkeypatch, version):
    async def agent(**kwargs):
        raise EnvironmentFailure("sandbox unavailable")

    tracer = _Tracer()
    wrapper = _patch(monkeypatch, agent, tracer)
    input = _input(version=version)

    output = await wrapper.generate(input)

    sample = output.samples[0] if version == "v2" else output.samples
    assert sample.status == Sample.Status.ABORTED
    assert sample is not input.sample
    assert tracer.collected == [(input.sample, 64, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_programming_error_fails_after_collection_even_if_collection_also_rejects(monkeypatch, version):
    async def agent(**kwargs):
        raise ValueError("bad agent implementation")

    tracer = _Tracer(error=HTTPResponseError("http://session/samples", 422, "invalid turns"))
    wrapper = _patch(monkeypatch, agent, tracer)

    with pytest.raises(ValueError, match="bad agent implementation"):
        await wrapper.generate(_input(version=version))

    assert len(tracer.collected) == 1


@pytest.mark.asyncio
async def test_legacy_mode_preserves_partial_sample_after_agent_exception(monkeypatch):
    async def agent(**kwargs):
        raise ValueError("legacy agent error")

    tracer = _Tracer()
    wrapper = _patch(monkeypatch, agent, tracer)

    output = await wrapper.generate(_input(version="v1", strict=False))

    assert output.samples.status == Sample.Status.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize(("status", "expected_abort"), [(503, True), (429, True), (422, False)])
async def test_collector_classifies_transient_and_structural_errors(monkeypatch, status, expected_abort):
    async def agent(**kwargs):
        return {"reward": 1.0}

    tracer = _Tracer(error=HTTPResponseError("http://session/samples", status, "failed"))
    wrapper = _patch(monkeypatch, agent, tracer)
    if expected_abort:
        output = await wrapper.generate(_input(version="v2"))
        assert output.samples[0].status == Sample.Status.ABORTED
    else:
        with pytest.raises(HTTPResponseError):
            await wrapper.generate(_input(version="v2"))


@pytest.mark.asyncio
async def test_v1_passes_score_identity_to_collector_and_protects_its_evidence(monkeypatch):
    async def agent(**kwargs):
        return {
            "reward": 0.8,
            "reward_record_id": "response-2",
            "stop_reason": "final_answer",
            "agent_metrics": {"total_tool_time": 2.5},
            "reward_valid_for_retained_tokens": False,
        }

    collected = Sample(
        status=Sample.Status.COMPLETED,
        metadata={"reward_valid_for_retained_tokens": True},
    )
    tracer = _Tracer(samples=[collected])
    wrapper = _patch(monkeypatch, agent, tracer)

    output = await wrapper.generate(_input(version="v1"))

    assert tracer.collected[0][2]["reward_record_id"] == "response-2"
    assert output.samples.metadata["reward_valid_for_retained_tokens"] is True
    assert output.samples.non_generation_time == 2.5


@pytest.mark.asyncio
async def test_cancellation_deletes_session_without_collecting_samples(monkeypatch):
    started = asyncio.Event()

    async def agent(**kwargs):
        started.set()
        await asyncio.Event().wait()

    tracer = _Tracer()
    wrapper = _patch(monkeypatch, agent, tracer)
    task = asyncio.create_task(wrapper.generate(_input(version="v1")))
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert tracer.collected == []
    assert tracer.deleted == [10]
