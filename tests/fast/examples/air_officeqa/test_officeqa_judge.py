"""OfficeQA path-report reward: judge pool routing and failover, record rebuild, reward mapping."""

import json

import httpx
import pytest

from examples.air_officeqa import judges, path_reward

_SUPPORTED = '{"path_status": "supported", "path_score": 0.8, "issues": []}'
_ENDPOINTS = ("judge-a", "judge-b", "judge-c")


class _Config:
    host = "https://workspace.example"

    def authenticate(self):
        return {"Authorization": "Bearer test"}


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _pool(handler, *, clock=None, temperature=None, primary_first=False):
    clock = clock or _Clock()
    settings = judges.PoolSettings(
        endpoints=_ENDPOINTS,
        max_concurrency=2,
        max_tokens=256,
        request_timeout_s=5,
        deadline_s=60,
        slot_wait_s=1,
        cooldown_s=10,
        temperature=temperature,
        primary_first=primary_first,
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return judges.JudgePool(settings, config=_Config(), client=client, clock=clock, sleep=clock.sleep)


def _reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": text}}]})


def _endpoint(request):
    return request.url.path.split("/")[2]


def test_a_prompt_key_keeps_its_judge_and_keys_spread_over_the_pool():
    pool = _pool(lambda request: _reply(_endpoint(request)))

    first = {pool.judge("UID0001", "s", "u")[0] for _ in range(3)}
    spread = {pool.judge(f"UID{i:04d}", "s", "u")[0] for i in range(40)}

    assert len(first) == 1
    assert spread == set(_ENDPOINTS)


def test_primary_first_sends_every_prompt_to_the_first_endpoint_until_it_fails():
    down = set()

    def handler(request):
        if _endpoint(request) in down:
            return httpx.Response(503)
        return _reply(_endpoint(request))

    pool = _pool(handler, primary_first=True)
    judged = {pool.judge(f"UID{i:04d}", "s", "u")[0] for i in range(40)}
    down.add(_ENDPOINTS[0])
    failover, _ = pool.judge("UID0001", "s", "u")

    assert judged == {_ENDPOINTS[0]}
    assert failover == _ENDPOINTS[1]


@pytest.mark.parametrize("temperature", [None, 0.0])
def test_temperature_is_sent_only_when_configured(temperature):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return _reply(_SUPPORTED)

    _pool(handler, temperature=temperature).judge("UID0001", "s", "u")

    assert bodies[0].get("temperature", "unset") == ("unset" if temperature is None else 0.0)


def test_rate_limited_endpoint_cools_down_and_the_call_fails_over():
    calls = []

    def handler(request):
        calls.append(_endpoint(request))
        if _endpoint(request) == calls[0]:
            return httpx.Response(429, headers={"retry-after": "30"})
        return _reply(_SUPPORTED)

    pool = _pool(handler)
    endpoint, reply = pool.judge("UID0001", "s", "u")
    again, _ = pool.judge("UID0001", "s", "u")

    assert reply == _SUPPORTED and endpoint != calls[0]
    assert again == endpoint and calls.count(calls[0]) == 1  # still cooling down: not retried


def test_a_request_every_endpoint_rejects_fails_fast():
    calls = []

    def handler(request):
        calls.append(_endpoint(request))
        return httpx.Response(400, json={"error": "prompt is too long"})

    with pytest.raises(judges.JudgeUnavailable, match="rejected"):
        _pool(handler).judge("UID0001", "s", "u")
    assert sorted(calls) == sorted(_ENDPOINTS)


def test_endpoints_that_keep_failing_end_at_the_deadline():
    clock = _Clock()
    with pytest.raises(judges.JudgeUnavailable, match="within 60s"):
        _pool(lambda request: httpx.Response(503), clock=clock).judge("UID0001", "s", "u")
    assert clock.now >= 60


def test_reply_text_reads_content_blocks_and_tolerates_garbage():
    blocks = httpx.Response(200, json={"choices": [{"message": {"content": [{"type": "text", "text": "ok"}]}}]})
    assert judges._reply_text(blocks) == "ok"
    assert judges._reply_text(httpx.Response(200, text="not json")) == ""


def _episode(*, answer="2602", claim="National defense row, calendar 1940 total = 2,602"):
    grep = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "grep_documents",
            "arguments": json.dumps({"pattern": "National defense", "year": "1941"}),
        },
    }
    bad = {"id": "c2", "type": "function", "function": {"name": "read_document", "arguments": '{"path": "/etc"}'}}
    # Tool-call parsers often send the nested path list as a JSON string.
    report = {"answer": answer, "path": json.dumps([{"id": "s1", "observation": "obs_1", "claim": claim}])}
    submit = {"id": "c3", "type": "function", "function": {"name": "submit_report", "arguments": json.dumps(report)}}
    return [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "What were U.S. national defense expenditures in 1940?"},
        {"role": "assistant", "content": None, "tool_calls": [grep, bad]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "[observation obs_1]\ntreasury_bulletin_1941_01.txt:722: National defense ... total 2,602",
        },
        {"role": "tool", "tool_call_id": "c2", "content": '{"error": "read_document is missing arguments"}'},
        {"role": "assistant", "content": None, "tool_calls": [submit]},
    ]


_METADATA = {
    "uid": "UID0007",
    "question": "What were U.S. national defense expenditures in 1940?",
    "reference_answer": "2602",
}


class _FakePool:
    def __init__(self, reply=_SUPPORTED, error=None):
        self.reply, self.error, self.keys = reply, error, []

    def judge(self, key, system, user):
        self.keys.append(key)
        if self.error is not None:
            raise self.error
        return "fake-judge", self.reply


def test_the_record_rebuilds_delivered_observations_and_the_report():
    record = path_reward.episode_record(_episode(), _METADATA)

    [obs] = record["observations"]  # the error reply carries no observation id
    assert obs["observation_id"] == "obs_1" and obs["tool"] == "grep_documents"
    assert obs["args"] == {"pattern": "National defense", "year": "1941"}
    assert obs["delivered_text"].startswith("treasury_bulletin_1941_01.txt:722")
    assert (obs["generated_by_request"], obs["delivered_to_request"]) == (0, 1)
    assert json.loads(record["terminal_text"])["path"][0]["observation"] == "obs_1"


def _report_episode(path, *, answer="2602", compute_with_read=False):
    """A grep of the 1940 total (obs_1) and a compute summing it (obs_2), then the report.

    With ``compute_with_read`` the compute call shares the grep's request, so it was generated
    before the grep's result arrived.
    """
    grep = {"id": "c1", "type": "function", "function": {"name": "grep_documents", "arguments": "{}"}}
    compute = {
        "id": "c2",
        "type": "function",
        "function": {"name": "compute", "arguments": json.dumps({"code": "print(1559 + 1043)"})},
    }
    report = {"answer": answer, "path": path}
    submit = {"id": "c3", "type": "function", "function": {"name": "submit_report", "arguments": json.dumps(report)}}
    first = [grep, compute] if compute_with_read else [grep]
    messages = [
        {"role": "user", "content": _METADATA["question"]},
        {"role": "assistant", "content": None, "tool_calls": first},
    ]
    messages.append(
        {"role": "tool", "tool_call_id": "c1", "content": "[observation obs_1]\nNational defense 1,559 1,043"}
    )
    if not compute_with_read:
        messages.append({"role": "assistant", "content": None, "tool_calls": [compute]})
    messages.append({"role": "tool", "tool_call_id": "c2", "content": "[observation obs_2]\nOutput:\n2602"})
    messages.append({"role": "assistant", "content": None, "tool_calls": [submit]})
    return messages


_SUM_PATH = [
    {"id": "s1", "observation": "obs_1", "claim": "National defense parts 1,559 and 1,043"},
    {"id": "s2", "observation": "obs_2", "claim": "1,559 + 1,043 = 2,602", "depends_on": ["s1"]},
]


def test_a_correct_answer_with_a_supported_path_earns_one(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(path_reward, "_judge_pool", lambda: pool)

    scored = path_reward.score(_episode(), _METADATA)

    assert scored["reward"] == 1.0  # not the judge's graded 0.8, which is kept for the logs
    assert (scored["category"], scored["path_score"], scored["judge"]) == ("correct", pytest.approx(0.8), "fake-judge")
    assert pool.keys == ["UID0007"]  # routed by prompt, so a GRPO group shares one judge


@pytest.mark.parametrize(
    ("episode", "pool", "category"),
    [
        (_episode(answer="2700", claim="National defense total = 2,602"), _FakePool(error=AssertionError), "wrong"),
        (_episode(claim="National defense total = 3,999"), _FakePool(error=AssertionError), "fabricated"),
        (
            _episode(),
            _FakePool(reply='{"path_status": "unsupported", "path_score": 0, "issues": ["s1"]}'),
            "unsupported",
        ),
        (_episode(), _FakePool(error=judges.JudgeUnavailable("pool exhausted")), "unknown"),
    ],
    ids=["wrong-answer-skips-judge", "fabricated-value-skips-judge", "unsupported", "judge-unavailable"],
)
def test_anything_short_of_a_supported_correct_report_scores_zero(monkeypatch, episode, pool, category):
    monkeypatch.setattr(path_reward, "_judge_pool", lambda: pool)

    scored = path_reward.score(episode, _METADATA)

    assert (scored["reward"], scored["category"]) == (0.0, category)


@pytest.mark.parametrize(
    ("answer", "category"),
    [
        ("1,559 or 2,602 million", "extra_values"),
        ("from 1,559 in 1939 to 2,602", "extra_values"),
        ("total 1,559 (1940: 2,602)", "not_leading"),
    ],
    ids=["candidates", "working-first", "gold-only-in-the-explanation"],
)
def test_an_answer_that_does_not_lead_with_one_final_value_scores_zero_without_the_judge(
    monkeypatch, answer, category
):
    pool = _FakePool()
    monkeypatch.setattr(path_reward, "_judge_pool", lambda: pool)

    # Gold 2602 appears in each answer, which the OfficeQA matcher alone would accept.
    assert path_reward.score(_episode(answer=answer), _METADATA)["category"] == category
    assert pool.keys == []


@pytest.mark.parametrize(
    "answer",
    [
        "2,602",
        "$2,602 million (calendar 1940, Table 3)",
        "2,602 million dollars. Sum of 12 months",
        "About 2,602 — see s1",
    ],
)
def test_an_answer_that_leads_with_the_final_value_goes_to_the_judge(monkeypatch, answer):
    pool = _FakePool()
    monkeypatch.setattr(path_reward, "_judge_pool", lambda: pool)

    assert path_reward.score(_episode(answer=answer), _METADATA)["reward"] == 1.0


def test_a_compute_input_that_arrived_after_the_call_is_counted_not_rejected(monkeypatch):
    """Rejecting it taught the policy to stop declaring compute inputs, which skips the check."""
    monkeypatch.setattr(path_reward, "_judge_pool", lambda: _FakePool())

    scored = path_reward.score(_report_episode(_SUM_PATH, compute_with_read=True), _METADATA)

    assert (scored["reward"], scored["late_inputs"]) == (1.0, 1)
    assert (scored["steps"], scored["compute_steps"], scored["compute_steps_with_inputs"]) == (2, 1, 1)


def test_a_lookup_step_that_declares_inputs_still_needs_its_figure_delivered(monkeypatch):
    monkeypatch.setattr(path_reward, "_judge_pool", lambda: _FakePool(error=AssertionError))
    path = [*_SUM_PATH, {"id": "s3", "observation": "obs_1", "claim": "1940 total 3,999", "depends_on": ["s2"]}]

    assert path_reward.score(_report_episode(path), _METADATA)["category"] == "fabricated"


def test_a_missing_judge_pool_fails_closed(monkeypatch):
    def broken():
        raise ValueError("no Databricks credentials")

    monkeypatch.setattr(path_reward, "_judge_pool", broken)

    assert path_reward.score(_episode(), _METADATA)["reward"] == 0.0


def test_cited_observations_survive_the_history_bound(monkeypatch):
    monkeypatch.setattr(path_reward, "MAX_HISTORY_BYTES", 2_000)
    messages = [{"role": "user", "content": "q"}]
    for i in range(1, 7):
        call = {"id": f"c{i}", "type": "function", "function": {"name": "read_document", "arguments": "{}"}}
        messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"[observation obs_{i}]\n" + "x" * 900})
    messages.append({"role": "assistant", "content": "done", "tool_calls": []})
    record = path_reward.episode_record(messages, {"uid": "UID1"})
    report = path_reward.path_report.parse_report(
        json.dumps({"answer": "1", "path": [{"id": "s", "observation": "obs_6", "claim": "late evidence"}]})
    )
    ledger = path_reward.path_report.EpisodeLedger.from_record(record)

    request = path_reward.support_request(record, report, ledger)

    kept = [row["observation_id"] for row in request["tool_history"]]
    assert "obs_6" in kept and kept == sorted(kept, key=lambda oid: int(oid[4:]))
    assert request["limits"]["omitted_observations"] == 6 - len(kept) > 0
