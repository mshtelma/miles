"""The standalone server is the rollout engine minus training-only flags, and proves it from its own log."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.glm5_3_flash.officeqa.eval import server

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
# A training rollout engine's own command line (from its log), minus host/port/seed values.
_TRAINING_FLAGS = (
    "--trust-remote-code --model-path --host --port --device cuda --random-seed --nccl-port --dist-init-addr "
    "--tp-size 8 --ep-size 8 --skip-server-warmup --enable-draft-weights-cpu-backup --enable-metrics "
    "--enable-return-routed-experts --engine-info-bootstrap-port --kv-cache-dtype bfloat16 --mem-fraction-static 0.7 "
    "--max-running-requests 96 --chunked-prefill-size 8192 --reasoning-parser glm45 --tool-call-parser glm47 "
    "--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --moe-runner-backend triton --lora-use-virtual-experts"
).split()
_TRAINING_ONLY = {
    "--nccl-port",
    "--dist-init-addr",
    "--enable-draft-weights-cpu-backup",
    "--enable-return-routed-experts",
    "--engine-info-bootstrap-port",
}


def _r3_server_args() -> str:
    return (_FIXTURES / "glm53_rollout_engine_server_args.txt").read_text()


def test_argv_keeps_every_engine_flag_but_the_training_only_ones() -> None:
    argv = server.server_argv(Path("/tmp/miles-air/models/m"), host="10.0.0.2", port=30000, seed=1235)

    flags = [token for token in argv[3:] if token.startswith("--")]
    expected = [flag for flag in _TRAINING_FLAGS if flag.startswith("--") and flag not in _TRAINING_ONLY]
    assert [
        flag for flag in flags if flag not in {"--served-model-name", "--default-chat-template-kwargs"}
    ] == expected
    assert argv[:3] == [sys.executable, "-m", "sglang.launch_server"]
    kwargs = json.loads(argv[argv.index("--default-chat-template-kwargs") + 1])
    assert kwargs == {"clear_thinking": False, "enable_thinking": True}


def test_parser_reads_the_real_engine_server_args_line() -> None:
    parsed = server.read_server_args("[2026-10-03 13:18:20] " + _r3_server_args() + "\nlater log line\n")

    assert len(parsed) > 400
    assert parsed["tp_size"] == 8 and parsed["mem_fraction_static"] == 0.7
    assert parsed["model_loader_extra_config"] == "{}" and parsed["enable_return_routed_experts"] is True
    assert parsed["default_chat_template_kwargs"] is None


def test_the_training_engines_differ_only_in_the_chat_template_kwargs() -> None:
    mismatches = server.parity_mismatches(server.read_server_args(_r3_server_args()))

    assert len(mismatches) == 1 and mismatches[0].startswith("default_chat_template_kwargs")


def test_an_eval_server_with_the_training_kwargs_passes_and_a_drifted_one_fails() -> None:
    line = _r3_server_args().replace(
        "default_chat_template_kwargs=None",
        "default_chat_template_kwargs={'clear_thinking': False, 'enable_thinking': True}",
    )
    assert server.parity_mismatches(server.read_server_args(line)) == []

    drifted = line.replace("moe_runner_backend='triton'", "moe_runner_backend='auto'")
    assert server.parity_mismatches(server.read_server_args(drifted)) == [
        "moe_runner_backend: expected 'triton', got 'auto'"
    ]


def test_a_log_without_server_args_is_an_error() -> None:
    with pytest.raises(ValueError, match="no server_args"):
        server.read_server_args("starting...\n")


def test_the_readiness_probe_samples_like_the_eval(tmp_path: Path) -> None:
    # An early eval died here: a probe without top_p/top_k took the model's default top_p and crashed every rank.
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    received = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload: dict) -> None:
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            self._reply({})

        def do_POST(self) -> None:
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._reply({"choices": [{"index": 0}]})

        def log_message(self, *_args) -> None:
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        server.wait_ready(f"http://127.0.0.1:{httpd.server_port}", process, timeout_s=10, poll_s=0.01)
    finally:
        process.kill()
        httpd.shutdown()

    assert (received[0]["top_p"], received[0]["top_k"], received[0]["temperature"]) == (1.0, -1, 1.0)


def test_wait_ready_fails_fast_when_the_server_exits(tmp_path: Path) -> None:
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
    process.wait()

    with pytest.raises(RuntimeError, match="exited with code 3"):
        server.wait_ready("http://127.0.0.1:9", process, timeout_s=60, poll_s=0.01)
