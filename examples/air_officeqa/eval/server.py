"""Standalone SGLang launch with the training engine settings."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from examples.air_officeqa.models import Model

SERVED_MODEL_NAME = "officeqa-eval"
DEFAULT_PORT = 30000
# What the recipe exports to its engines (examples/air_officeqa/run_officeqa.py, _env_vars).
SERVER_ENV = {
    "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
    "SGLANG_RETURN_ORIGINAL_LOGPROB": "0",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
}
# Full-support sampling, as every training request sets it. A request that omits these gets the model's
# generation-config top_p (sampling_defaults='model'), which differs from training.
FULL_SUPPORT_SAMPLING = {"top_p": 1.0, "top_k": -1}


def chat_template(model: Model) -> tuple[str | None, dict]:
    """The training renderer's chat template, a bundled file or None for the checkpoint's own, and its kwargs."""
    # Imported on use, as below: the eval's modules import without the training stack (transformers, torch).
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    return resolve_fixed_chat_template(model.tito_model)


def server_argv(model: Model, model_path: Path, *, host: str, port: int, seed: int) -> list[str]:
    """The training rollout-engine argv minus training-only flags, plus the training chat template."""
    from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(model.tito_model)
    template, template_kwargs = chat_template(model)
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
        str(model.engine_gpus),
        "--ep-size",
        str(model.engine_gpus),
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
        reasoning_parser,
        "--tool-call-parser",
        tool_call_parser,
        *model.engine_args,
        "--lora-use-virtual-experts",
        *(["--chat-template", template] if template else []),
        "--default-chat-template-kwargs",
        json.dumps(template_kwargs),
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

    Loading the weights and compiling kernels takes minutes, so ``timeout_s`` must be generous.
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
