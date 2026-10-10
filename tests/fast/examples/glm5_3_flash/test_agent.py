"""GLM agent tests for raw TITO messages, tool boundaries, and terminal scoring."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from examples.glm5_3_flash import agent, reward
from miles.utils.types import Sample


def _mock_responses(monkeypatch, responses):
    requests = []
    response_iter = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=next(response_iter))

    client_type = httpx.AsyncClient
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(agent.httpx, "AsyncClient", lambda **kwargs: client_type(transport=transport, **kwargs))
    return requests


def _completion(response_id: str, *, calls=None, finish="stop") -> dict:
    return {
        "id": response_id,
        "choices": [
            {
                "message": {"role": "assistant", "content": None if calls else "done", "tool_calls": calls or []},
                "finish_reason": finish,
            }
        ],
        "usage": {"total_tokens": 20},
    }


def _call(arguments: str = "{}") -> dict:
    return {"id": "tool-1", "type": "function", "function": {"name": "search", "arguments": arguments}}


def _metadata() -> dict:
    return {
        "session_id": "trusted-session",
        "attempt_id": "trusted-attempt",
        "max_seq_len": 100,
        "tool_context": {"task_id": "task-7"},
        "tools": [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}],
        "reference_answer": "private scorer input",
    }


@pytest.mark.asyncio
async def test_agent_keeps_raw_roles_and_scores_only_the_terminal_record(monkeypatch):
    requests = _mock_responses(
        monkeypatch,
        [_completion("response-1", calls=[_call('{"query":"x"}')], finish="tool_calls"), _completion("response-2")],
    )
    tool_calls = []

    def run_tool(name, arguments, context):
        tool_calls.append((name, arguments, context))
        return "result"

    result = await agent._drive(
        "http://session",
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "question"}],
        {},
        _metadata(),
        agent.AgentSettings(max_turns=2, reserve_tokens=0),
        run_tool,
        lambda messages, metadata: 0.75,
    )

    assert result["reward"] == 0.75
    assert result["reward_record_id"] == "response-2"
    assert result["stop_reason"] == "final_answer"
    assert [message["role"] for message in requests[0]["messages"]] == ["system", "user"]
    assert [message["role"] for message in requests[1]["messages"]] == ["system", "user", "assistant", "tool"]
    assert tool_calls == [
        (
            "search",
            {"query": "x"},
            {
                "task": {"task_id": "task-7"},
                "session_ref": hashlib.sha256(b"trusted-session\x00trusted-attempt").hexdigest(),
                "attempt_id": "trusted-attempt",
                "tool_call_id": "tool-1",
                "tool_deadline_s": 600.0,
            },
        )
    ]


@pytest.mark.asyncio
async def test_malformed_policy_tool_arguments_are_returned_as_tool_text():
    def unexpected_tool(*args):
        raise AssertionError("invalid policy arguments reached the sandbox")

    text = await agent._tool_result(
        unexpected_tool,
        call=_call('{"query": NaN}'),
        metadata=_metadata(),
        allowed_tools={"search"},
        settings=agent.AgentSettings(),
    )

    assert "non-JSON numeric constant" in json.loads(text)["error"]


@pytest.mark.asyncio
async def test_max_turns_does_not_score_an_unfinished_tool_trajectory(monkeypatch):
    _mock_responses(
        monkeypatch,
        [
            _completion("response-1", calls=[_call()], finish="tool_calls"),
            _completion("response-2", calls=[_call()], finish="tool_calls"),
        ],
    )

    def unexpected_score(*args):
        raise AssertionError("unfinished trajectory reached the scorer")

    result = await agent._drive(
        "http://session",
        "question",
        {},
        _metadata(),
        agent.AgentSettings(max_turns=2, reserve_tokens=0),
        lambda name, arguments, context: "ok",
        unexpected_score,
    )

    assert result["stop_reason"] == "max_turns"
    assert result["reward"] == 0.0
    assert result["num_turns"] == 2


@pytest.mark.asyncio
async def test_terminal_tool_call_is_scored_unexecuted_even_near_the_budget(monkeypatch):
    submit = {"id": "tool-9", "type": "function", "function": {"name": "submit", "arguments": '{"answer":"42"}'}}
    _mock_responses(monkeypatch, [_completion("response-1", calls=[submit], finish="tool_calls")])
    metadata = _metadata()
    metadata["tools"].append({"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}})
    scored = []

    def unexpected_tool(*args):
        raise AssertionError("a terminal tool call reached the sandbox")

    def score(messages, metadata):
        scored.append(messages[-1]["tool_calls"][0]["function"]["arguments"])
        return 1.0

    result = await agent._drive(
        "http://session",
        "question",
        {},
        metadata,
        # 20 used + 90 reserved exceeds the 100-token budget, which would otherwise stop as "budget".
        agent.AgentSettings(max_turns=4, reserve_tokens=90, terminal_tools=("submit",)),
        unexpected_tool,
        score,
    )

    assert result["stop_reason"] == "final_answer"
    assert result["reward"] == 1.0
    assert result["reward_record_id"] == "response-1"
    assert scored == ['{"answer":"42"}']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scored", "reward", "details"),
    [(0.5, 0.5, None), ({"reward": 1.0, "category": "correct"}, 1.0, {"category": "correct"})],
    ids=["plain-reward", "reward-with-details"],
)
async def test_a_scorer_may_return_the_details_behind_its_reward(monkeypatch, scored, reward, details):
    _mock_responses(monkeypatch, [_completion("response-1")])

    result = await agent._drive(
        "http://session", "question", {}, _metadata(), agent.AgentSettings(), None, lambda messages, metadata: scored
    )

    assert result["reward"] == reward
    assert result.get("reward_details") == details


@pytest.mark.asyncio
@pytest.mark.parametrize("scored", [{"category": "correct"}, {"reward": float("nan")}, True])
async def test_a_scorer_without_a_finite_reward_fails(monkeypatch, scored):
    _mock_responses(monkeypatch, [_completion("response-1")])

    with pytest.raises(ValueError, match="non-finite reward"):
        await agent._drive(
            "http://session",
            "question",
            {},
            _metadata(),
            agent.AgentSettings(),
            None,
            lambda messages, metadata: scored,
        )


def test_terminal_tools_accept_a_json_list_and_reject_bad_names():
    settings = agent.AgentSettings(**json.loads('{"terminal_tools": ["submit_report"]}'))
    assert settings.terminal_tools == ("submit_report",)
    for bad in ("submit_report", ["submit_report", "submit_report"], [""], [7]):
        with pytest.raises(ValueError, match="terminal_tools"):
            agent.AgentSettings(terminal_tools=bad)


def test_finish_reason_must_match_the_assistant_tool_calls():
    with pytest.raises(ValueError, match="disagree"):
        agent._stop_reason(
            _completion("response-1", finish="tool_calls")["choices"][0],
            used_tokens=None,
            budget=None,
            reserve=0,
        )


@pytest.mark.asyncio
async def test_reward_is_zero_for_a_discarded_final_answer():
    sample = Sample(metadata={"reward": 0.8, "reward_valid_for_retained_tokens": True})

    assert await reward.reward_func(SimpleNamespace(), sample) == pytest.approx(0.8)
    assert sample.metadata["raw_reward"] == 0.8
    sample.metadata["reward_valid_for_retained_tokens"] = False
    assert await reward.reward_func(SimpleNamespace(), sample) == 0.0
    assert sample.metadata["raw_reward"] == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments,message",
    [
        ('{"n": ' + "1" * 5000 + "}", "ValueError"),  # beyond Python's int-digit limit
        ('{"q": ' + "[" * 100_000 + "]" * 100_000 + "}", "RecursionError"),
        ('{"x": 1e999}', "non-finite"),
        ('{"q": ', "JSONDecodeError"),
    ],
)
async def test_pathological_policy_arguments_stay_policy_visible(arguments, message):
    def unexpected_tool(*args):
        raise AssertionError("invalid policy arguments reached the sandbox")

    text = await agent._tool_result(
        unexpected_tool,
        call=_call(arguments),
        metadata=_metadata(),
        allowed_tools={"search"},
        settings=agent.AgentSettings(),
    )

    assert message in json.loads(text)["error"]


@pytest.mark.asyncio
async def test_tool_text_with_lone_surrogates_is_returned_not_raised():
    raw = b"ok \xff done".decode("utf-8", errors="surrogateescape")
    text = await agent._tool_result(
        lambda *args: raw,
        call=_call('{"query": "x"}'),
        metadata=_metadata(),
        allowed_tools={"search"},
        settings=agent.AgentSettings(),
    )

    assert text.startswith("ok ") and text.endswith(" done")


@pytest.mark.asyncio
async def test_tool_uses_its_own_executor(monkeypatch):
    async def forbidden_default_pool(*args):
        raise AssertionError("tool call reached the shared thread pool")

    monkeypatch.setattr(agent.asyncio, "to_thread", forbidden_default_pool)
    text = await agent._tool_result(
        lambda *args: "ok",
        call=_call(),
        metadata=_metadata(),
        allowed_tools={"search"},
        settings=agent.AgentSettings(),
    )
    assert text == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 404, 429, 500])
async def test_model_errors_include_bounded_server_detail(status):
    async def handler(request):
        return httpx.Response(status, json={"error": {"message": "session rejected " + "x" * 1000}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(agent.EnvironmentFailure, match=f"model status {status}: .*session rejected") as error:
            await agent._complete(client, "http://session", messages=[], tools=[], request={})
    assert len(str(error.value)) <= 540


def _flaky_handler(failures: list[Exception]):
    attempts = []

    async def handler(request):
        attempts.append(json.loads(request.content))
        if failures:
            raise failures.pop(0)
        return httpx.Response(200, json=_completion("response-1"))

    return handler, attempts


@pytest.mark.asyncio
async def test_a_dropped_connection_resends_the_same_request(monkeypatch):
    async def no_sleep(delay):
        pass

    monkeypatch.setattr(agent.asyncio, "sleep", no_sleep)
    handler, attempts = _flaky_handler([httpx.ReadError("closed"), httpx.RemoteProtocolError("closed")])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        choice, _, record_id, retries = await agent._complete(
            client, "http://session", messages=[{"role": "user", "content": "q"}], tools=[], request={}
        )
    assert (choice["message"]["content"], record_id, retries) == ("done", "response-1", 2)
    assert len(attempts) == 3 and attempts[0] == attempts[1] == attempts[2]


@pytest.mark.asyncio
async def test_connection_drops_past_the_retry_budget_fail_the_trajectory(monkeypatch):
    async def no_sleep(delay):
        pass

    monkeypatch.setattr(agent.asyncio, "sleep", no_sleep)
    handler, attempts = _flaky_handler([httpx.ReadError("closed")] * 3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(agent.EnvironmentFailure, match="model transport failure"):
            await agent._complete(client, "http://session", messages=[], tools=[], request={})
    assert len(attempts) == 3


@pytest.mark.asyncio
async def test_a_read_timeout_is_not_resent():
    handler, attempts = _flaky_handler([httpx.ReadTimeout("slow")])
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(agent.EnvironmentFailure, match="model transport failure"):
            await agent._complete(client, "http://session", messages=[], tools=[], request={})
    assert len(attempts) == 1
