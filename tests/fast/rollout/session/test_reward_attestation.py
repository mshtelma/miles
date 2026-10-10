"""The v1 session collector attests whether a scored final answer survived truncation."""

import copy
import importlib
from types import SimpleNamespace

import pytest

from miles.utils.types import Sample


def _turns() -> list[Sample]:
    return [
        Sample(
            prompt="prompt",
            tokens=[1, 2, 10],
            response_length=1,
            response="a",
            loss_mask=[1],
            rollout_log_probs=[-0.2],
            status=Sample.Status.COMPLETED,
        ),
        Sample(
            prompt="prompt with result",
            tokens=[1, 2, 10, 99, 20],
            response_length=1,
            response="b",
            loss_mask=[1],
            rollout_log_probs=[-0.3],
            status=Sample.Status.COMPLETED,
        ),
    ]


def _core(monkeypatch):
    session_core = importlib.import_module("miles.rollout.session.core")
    session = SimpleNamespace(
        records=[SimpleNamespace(response={"id": "response-1"}), SimpleNamespace(response={"id": "response-2"})],
        max_seq_len=8,
        sampling_support_replay=False,
    )
    core = object.__new__(session_core.SessionCore)
    core.registry = SimpleNamespace(
        get_session=lambda session_id: session, tokenizer=SimpleNamespace(decode=lambda tokens: str(tokens))
    )
    core.config = SimpleNamespace(
        save_debug_trajectory_data=None,
        sglang_speculative_algorithm=None,
        session_capture_provenance=True,
    )
    core.use_addition_r3 = False
    core._session_metadata = lambda session_id, session: {}
    monkeypatch.setattr(
        session_core, "compute_samples_from_openai_records", lambda *args, **kwargs: copy.deepcopy(_turns())
    )
    captured = []

    def fake_encode(samples, metadata, **kwargs):
        captured.append(samples)
        return b"samples"

    monkeypatch.setattr(session_core, "encode_samples", fake_encode)
    return core, captured


@pytest.mark.asyncio
async def test_v1_attests_only_the_retained_terminal_response(monkeypatch):
    core, captured = _core(monkeypatch)
    agent = {
        "reward": 0.9,
        "reward_record_id": "response-2",
        "stop_reason": "final_answer",
        "reward_valid_for_retained_tokens": False,
    }

    response = await core.collect_samples("sid", max_seq_len=5, agent_metadata=agent)

    assert response.status_code == 200
    (sample,) = captured[-1]
    assert sample.status == Sample.Status.COMPLETED
    assert sample.tokens == [1, 2, 10, 99, 20]
    assert sample.loss_mask == [1, 0, 1]
    assert sample.metadata["retained_terminal_record_id"] == "response-2"
    assert sample.metadata["reward_valid_for_retained_tokens"] is True


@pytest.mark.asyncio
async def test_v1_dropped_final_turn_invalidates_its_reward(monkeypatch):
    core, captured = _core(monkeypatch)

    response = await core.collect_samples(
        "sid",
        max_seq_len=4,
        agent_metadata={"reward": 0.9, "reward_record_id": "response-2", "stop_reason": "final_answer"},
    )

    assert response.status_code == 200
    (sample,) = captured[-1]
    assert sample.status == Sample.Status.TRUNCATED
    assert sample.tokens == [1, 2, 10]
    assert sample.metadata["original_token_count"] == 5
    assert sample.metadata["retained_token_count"] == 3
    assert sample.metadata["removed_turn_count"] == 1
    assert sample.metadata["retained_terminal_record_id"] is None
    assert sample.metadata["reward_valid_for_retained_tokens"] is False


@pytest.mark.asyncio
async def test_v1_all_turns_dropped_has_no_trainable_sample(monkeypatch):
    core, captured = _core(monkeypatch)

    response = await core.collect_samples("sid", max_seq_len=2, agent_metadata=None)

    assert response.status_code == 200
    assert captured[-1] == []
