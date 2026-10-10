"""Standalone SGLang launch with the training engine settings."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

# The training renderer's kwargs (miles/utils/chat_template_utils/tito_tokenizer.py, GLM53TITOTokenizer).
TEMPLATE_KWARGS = {"clear_thinking": False, "enable_thinking": True}
SERVED_MODEL_NAME = "glm53-eval"
DEFAULT_PORT = 30000
# What the recipe exports to its engines (scripts/run_glm5_3_flash.py, _env_vars).
SERVER_ENV = {
    "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
    "SGLANG_RETURN_ORIGINAL_LOGPROB": "0",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}
# Full-support sampling, as every training request sets it. A request that omits these gets the model's
# generation-config top_p (sampling_defaults='model'), which differs from training and, on this image,
# JIT-compiles a flashinfer kernel that fails for a missing curand.h and crashes every TP rank.
FULL_SUPPORT_SAMPLING = {"top_p": 1.0, "top_k": -1}
# GLM-5.3's stop tokens (generation_config.json eos_token_id 154829, 154827, 154820). With no_stop_trim
# the one that ended a turn stays at the end of the returned text.
STOP_TEXTS = ("<|observation|>", "<|user|>", "<|endoftext|>")


def server_argv(model_path: Path, *, host: str, port: int, seed: int) -> list[str]:
    """The training rollout-engine argv minus training-only flags, plus the training chat-template kwargs."""
    return [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--trust-remote-code",
        "--model-path",
        str(model_path),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        host,
        "--port",
        str(port),
        "--device",
        "cuda",
        "--random-seed",
        str(seed),
        "--tp-size",
        "8",
        "--ep-size",
        "8",
        "--skip-server-warmup",
        "--enable-metrics",
        "--kv-cache-dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "96",
        "--chunked-prefill-size",
        "8192",
        "--reasoning-parser",
        "glm45",
        "--tool-call-parser",
        "glm47",
        "--dsa-prefill-backend",
        "tilelang",
        "--dsa-decode-backend",
        "tilelang",
        "--moe-runner-backend",
        "triton",
        "--lora-use-virtual-experts",
        "--default-chat-template-kwargs",
        json.dumps(TEMPLATE_KWARGS),
    ]


def launch(argv: list[str], *, log_path: Path) -> subprocess.Popen:
    """Start the server in its own process group, writing its combined output to ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        return subprocess.Popen(
            argv,
            env={**os.environ, **SERVER_ENV},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def wait_ready(url: str, process: subprocess.Popen, *, timeout_s: float, poll_s: float = 10.0) -> float:
    """Block until the server answers a one-token chat completion; fail fast if it exits.

    Loading 328 GB of weights and compiling kernels takes minutes, so ``timeout_s`` must be generous.
    Returns the seconds it took.
    """
    started = time.monotonic()
    probe = {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 1.0,
        **FULL_SUPPORT_SAMPLING,  # the eval's sampling path, not the model's default
    }
    while True:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {process.returncode} before becoming ready")
        try:
            with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
                if client.get(f"{url}/health").status_code == 200:
                    response = client.post(f"{url}/v1/chat/completions", json=probe)
                    if response.status_code == 200 and response.json().get("choices"):
                        return time.monotonic() - started
        except (httpx.HTTPError, ValueError):
            pass
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(f"SGLang server at {url} was not ready after {timeout_s:.0f} s")
        time.sleep(poll_s)


def stop(process: subprocess.Popen | None) -> None:
    """Stop the server's process group: SIGTERM, then SIGKILL after 30 s."""
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
