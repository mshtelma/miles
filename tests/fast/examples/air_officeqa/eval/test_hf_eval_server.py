"""The standalone server is the training rollout engine minus its training-only flags."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from examples.air_officeqa.eval import server
from examples.air_officeqa.models import MODELS

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


def test_argv_keeps_every_engine_flag_but_the_training_only_ones() -> None:
    argv = server.server_argv(
        MODELS["GLM-5.3-Flash"], Path("/tmp/miles-air/models/m"), host="10.0.0.2", port=30000, seed=1235
    )

    flags = [token for token in argv[3:] if token.startswith("--")]
    expected = [flag for flag in _TRAINING_FLAGS if flag.startswith("--") and flag not in _TRAINING_ONLY]
    assert [
        flag for flag in flags if flag not in {"--served-model-name", "--default-chat-template-kwargs"}
    ] == expected
    assert argv[:3] == [sys.executable, "-m", "sglang.launch_server"]
    kwargs = json.loads(argv[argv.index("--default-chat-template-kwargs") + 1])
    assert kwargs == {"clear_thinking": False, "enable_thinking": True}
    assert "--chat-template" not in argv  # GLM-5.3's renderer keeps the checkpoint's own template


def test_qwen35_is_served_with_its_renderers_template_and_parsers() -> None:
    argv = server.server_argv(MODELS["Qwen3.5-35B-A3B"], Path("/tmp/m"), host="10.0.0.2", port=30000, seed=1)

    def value(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    assert (value("--reasoning-parser"), value("--tool-call-parser")) == ("qwen3", "qwen3_coder")
    assert Path(value("--chat-template")).name == "qwen3.5_fixed.jinja"
    assert json.loads(value("--default-chat-template-kwargs")) == {"preserve_thinking": True}
    assert value("--attention-backend") == "flashinfer" and "--dsa-prefill-backend" not in argv


def test_the_readiness_probe_samples_like_the_eval(tmp_path: Path) -> None:
    # Without top_p and top_k, the probe gets the model's default top_p, which crashes every rank.
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
