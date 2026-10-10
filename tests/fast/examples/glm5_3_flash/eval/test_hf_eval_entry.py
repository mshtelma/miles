import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.glm5_3_flash.officeqa.eval import entry
from tools.air import process, stage_models


@pytest.fixture(autouse=True)
def fast_polls(monkeypatch):
    monkeypatch.setattr(entry, "POLL_S", 0.01)


def _register(root, rank, identity=None):
    path = root / "servers" / f"node_{rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        stage_models.json_bytes({"url": f"http://10.0.0.{rank}:30000", "model": identity or {"kind": "base"}})
    )


def _sleeper():
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)


def test_servers_register_in_rank_order(tmp_path):
    _register(tmp_path, 1)
    _register(tmp_path, 0)
    assert entry._wait_for_servers(tmp_path, 2, {"kind": "base"}, deadline=time.monotonic() + 1) == [
        "http://10.0.0.0:30000",
        "http://10.0.0.1:30000",
    ]


def test_servers_must_have_the_same_model(tmp_path):
    _register(tmp_path, 0, {"kind": "checkpoint"})
    with pytest.raises(ValueError, match="different models"):
        entry._wait_for_servers(tmp_path, 1, {"kind": "base"}, deadline=time.monotonic() + 1)


def test_peer_failure_stops_registration(tmp_path):
    _register(tmp_path, 0)
    entry._write_failure(tmp_path, 1, RuntimeError("engine died"))
    with pytest.raises(RuntimeError, match="failed before serving"):
        entry._wait_for_servers(tmp_path, 2, {"kind": "base"}, deadline=time.monotonic() + 1)
    assert stage_models.read_json(tmp_path / "failures/node_1.json")["cause"] == "engine died"


def test_missing_server_times_out(tmp_path):
    _register(tmp_path, 0)
    with pytest.raises(TimeoutError, match="1 of 2"):
        entry._wait_for_servers(tmp_path, 2, {"kind": "base"}, deadline=time.monotonic() + 0.03)


@pytest.mark.parametrize("code", [0, 7])
def test_worker_returns_the_head_outcome(tmp_path, code):
    (tmp_path / "result.json").write_text(json.dumps({"exit_code": code}))
    child = _sleeper()
    try:
        assert entry._serve_until_done(tmp_path, child, deadline=time.monotonic() + 1) == code
    finally:
        process.stop_process(child)


def test_worker_reports_local_engine_exit_and_peer_failure(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], start_new_session=True)
    child.wait()
    with pytest.raises(RuntimeError, match="exited with code 3"):
        entry._serve_until_done(tmp_path, child, deadline=time.monotonic() + 1)
    entry._write_failure(tmp_path, 0, ValueError("head failed"))
    child = _sleeper()
    try:
        with pytest.raises(RuntimeError, match="another eval node failed"):
            entry._serve_until_done(tmp_path, child, deadline=time.monotonic() + 1)
    finally:
        process.stop_process(child)


def test_head_engine_exit_is_detected_while_waiting_for_peers(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"], start_new_session=True)
    child.wait()
    with pytest.raises(RuntimeError, match="head eval engine exited"):
        entry._wait_for_servers(tmp_path, 2, {}, deadline=time.monotonic() + 10, process=child)


def test_failed_readiness_stops_the_server(tmp_path, monkeypatch):
    child = _sleeper()
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setattr(entry, "_routable_ip", lambda address: "127.0.0.1")
    monkeypatch.setattr(stage_models, "stage_hf", lambda *args, **kwargs: SimpleNamespace(path=tmp_path, identity={}))
    monkeypatch.setattr(entry.server, "launch", lambda *args, **kwargs: child)

    def not_ready(*args, **kwargs):
        raise TimeoutError("not ready")

    monkeypatch.setattr(entry.server, "wait_ready", not_ready)
    args = SimpleNamespace(
        state_root=tmp_path,
        input_root=tmp_path,
        root=tmp_path / "volume",
        manifest=tmp_path / "manifest.json",
        num_nodes=1,
        rank=0,
        hf_source="base",
        port=30000,
    )
    try:
        with pytest.raises(TimeoutError, match="not ready"):
            entry._start_server(args, tmp_path / "server.log", time.monotonic() + 1)
        assert child.poll() is not None
    finally:
        process.stop_process(child)


def test_eval_set_paths_are_relative_to_staged_inputs(tmp_path):
    assert entry._eval_sets(["heldout=data/officeqa/eval.jsonl"], tmp_path) == {
        "heldout": tmp_path / "data/officeqa/eval.jsonl"
    }
    with pytest.raises(ValueError, match="unique"):
        entry._eval_sets(["x=/tmp/a", "x=/tmp/b"], tmp_path)


def test_routable_ip_uses_the_route_to_the_head():
    assert entry._routable_ip("127.0.0.1") == "127.0.0.1"


async def test_peer_engine_failure_interrupts_scoring(tmp_path, monkeypatch):
    import examples.glm5_3_flash.agent as agent

    async def suspended(**kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(agent, "run", suspended)
    monkeypatch.setattr(entry, "_token_counter", lambda path: lambda messages, tools: 0)

    async def cleanup():
        pass

    async def serve(chat):
        return SimpleNamespace(cleanup=cleanup), "http://127.0.0.1:9"

    async def local_io(function, *args):
        return function(*args)

    monkeypatch.setattr(entry.proxy, "serve", serve)
    monkeypatch.setattr(entry.asyncio, "to_thread", local_io)
    row = tmp_path / "heldout.jsonl"
    row.write_text(json.dumps({"prompt": "q", "metadata": {"uid": "q1", "tools": [], "tool_context": {}}}) + "\n")
    entry._write_failure(tmp_path, 1, RuntimeError("peer engine died"))
    child = _sleeper()
    args = SimpleNamespace(
        root=tmp_path,
        max_seq_len=150_000,
        capture_trajectories=0,
        limit_questions=0,
        eval_sets={"heldout": row},
        samples_per_prompt=1,
        max_concurrent=1,
        attempt_id="attempt",
    )
    try:
        with pytest.raises(RuntimeError, match="engines failed during scoring"):
            await entry._evaluate(args, ["http://127.0.0.1:9"], tmp_path, child, time.monotonic() + 1)
    finally:
        process.stop_process(child)


def test_the_token_counter_renders_a_prior_tool_call_the_way_sglang_does(tmp_path) -> None:
    # An eval once lost every trajectory here: the agent sends tool-call arguments as a JSON string, and the GLM
    # template iterates them as a dict (chat_template.jinja: ``tc.arguments.items()``).
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    words.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=words, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% for tc in m.tool_calls or [] %}"
        "{% for k, v in tc.function.arguments.items() %}{{ k }} {{ v }} {% endfor %}{% endfor %}{% endfor %}"
    )
    tokenizer.save_pretrained(tmp_path)
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "obs"},
    ]

    count = entry._token_counter(tmp_path)

    assert count(messages, None) == len("user q assistant query x tool obs".split())


def test_eval_imports_without_training_or_model_dependencies():
    script = (
        "import sys\n"
        "for name in ('torch', 'ray', 'megatron', 'sglang', 'transformers'):\n"
        "    sys.modules[name] = None\n"
        "import examples.glm5_3_flash.officeqa.eval.entry\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[5])
