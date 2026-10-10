"""One AIR node supervisor: Ray bootstrap, input staging and launcher lifecycle."""

import argparse
import asyncio
import logging
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import ray
from tools.air import fuse_transfer as fuse
from tools.air import publisher, resume, stage_models
from tools.air.process import LogPublisher, read_result, resolve_path, signal_exit, stop_process

from miles.utils.function_registry import load_function

RAY_PORT = 6379
POLL_S = 10
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Launch:
    command: list[str]
    training_args: list[str]
    model_name: str
    reference_name: str
    save_hf: bool


def _tee(source, sink):
    for chunk in iter(lambda: source.read1(64 * 1024), b""):
        sink.write(chunk)
        sink.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()


def prepare_launch(args, launcher_args: list[str]) -> tuple[Launch, dict]:
    manifest_path = resolve_path(args.input_manifest, args.volume_root / "inputs/latest")
    inputs = stage_models.load_inputs(manifest_path)
    reference_root = resolve_path(args.reference_root, args.volume_root / "ckpt/latest")
    reference = stage_models.read_reference(reference_root, manifest_path.stem)
    latest = resume.latest_complete(args.volume_root, args.run_id)
    args.input_root = args.local_root / "inputs" / manifest_path.stem
    args.state_root = args.local_root / "runs" / args.run_id / args.attempt_id
    resume_root = args.state_root / "dcp/resume" if latest is not None else None
    launch = load_function(args.launcher_path)(args, launcher_args, resume_root=resume_root)
    configuration = resume.configuration(
        launch.training_args, inputs, launch.model_name, manifest_path.stem, reference
    )
    if latest is not None:
        resume.check_compatibility(latest[1], configuration)
    sources = [
        {
            "remote_root": str(reference_root),
            "local_root": str(args.state_root / "dcp/reference" / launch.reference_name),
            "document": reference,
        }
    ]
    if latest is not None:
        sources.append(
            {
                "remote_root": str(latest[0]),
                "local_root": str(resume_root),
                "document": latest[1],
            }
        )
    config = {
        "volume_root": str(args.volume_root),
        "run_id": args.run_id,
        "attempt_id": args.attempt_id,
        "state_root": str(args.state_root),
        "slot_root": str(args.local_root / "transfer-slots"),
        "checkpoint_root": str(args.state_root / "checkpoints"),
        "input_manifest": str(manifest_path),
        "input_root": str(args.input_root),
        "model_name": launch.model_name,
        "sources": sources,
        "configuration": configuration,
        "workers": max(1, 32 // args.num_nodes),
        "save_hf": launch.save_hf,
        "reference_bytes": sum(f["size_bytes"] for f in reference["files"]),
        "hf_bytes": sum(f["size_bytes"] for f in stage_models.model_record(inputs, launch.model_name)["files"]),
    }
    return launch, config


def stage_node(config: dict) -> None:
    state = Path(config["state_root"])
    state.mkdir(parents=True, exist_ok=True)
    limiter = fuse.NodeTransferLimiter(Path(config["slot_root"]), config["workers"])
    inputs = stage_models.load_inputs(Path(config["input_manifest"]))
    stage_models.stage_inputs(inputs, Path(config["input_root"]), config["model_name"], limiter)
    for source in config["sources"]:
        stage_models.stage_checkpoint(
            Path(source["remote_root"]), Path(source["local_root"]), source["document"], limiter
        )
    (state / "runtime.json").write_bytes(stage_models.json_bytes(config))
    print(f"AIR inputs staged on {socket.gethostname()}", flush=True)


def ray_argv(args, *, head: bool) -> list[str]:
    flags = ["ray", "start", f"--num-gpus={args.num_gpus_per_node}", "--disable-usage-stats"]
    if head:
        return [
            *flags,
            "--head",
            f"--node-ip-address={args.master_addr}",
            f"--port={RAY_PORT}",
            "--dashboard-host=127.0.0.1",
        ]
    return [*flags, f"--address={args.master_addr}:{RAY_PORT}"]


def launcher_env() -> dict[str, str]:
    return {
        **os.environ,
        "MILES_SCRIPT_EXTERNAL_RAY": "1",
        "RAY_ADDRESS": "http://127.0.0.1:8265",
        "PYTHONUNBUFFERED": "1",
    }


def _wait_nodes(args) -> list[str]:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        nodes = [n for n in ray.nodes() if n["Alive"]]
        if len(nodes) > args.num_nodes:
            raise ValueError("Ray has more nodes than the AIR allocation")
        if len(nodes) == args.num_nodes:
            if any(n["Resources"].get("GPU", 0) != args.num_gpus_per_node for n in nodes):
                raise ValueError("Ray GPU resources differ from the AIR allocation")
            return sorted(n["NodeID"] for n in nodes)
        time.sleep(2)
    raise TimeoutError("AIR nodes did not join Ray")


def run_head(args, launcher_args: list[str], result_root: Path) -> int:
    launch, config = prepare_launch(args, launcher_args)
    subprocess.run(ray_argv(args, head=True), check=True, timeout=120)
    ray.init(address=f"{args.master_addr}:{RAY_PORT}", logging_level="ERROR")
    config["node_ids"] = _wait_nodes(args)
    asyncio.run(publisher.on_nodes(stage_node, config["node_ids"], config, timeout_s=1800))
    log_path = args.state_root / "launcher.log"
    with log_path.open("wb") as log, LogPublisher(log_path, result_root / "logs/launcher"):
        child = subprocess.Popen(
            launch.command,
            env=launcher_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        tee = threading.Thread(target=_tee, args=(child.stdout, log), daemon=True)
        tee.start()
        deadline = time.monotonic() + args.timeout_s
        try:
            while child.poll() is None:
                alive = {node["NodeID"] for node in ray.nodes() if node["Alive"]}
                if not set(config["node_ids"]) <= alive:
                    raise RuntimeError("an AIR Ray node disappeared during training")
                if time.monotonic() >= deadline:
                    raise TimeoutError("AIR launcher exceeded its runtime budget")
                time.sleep(2)
            return child.returncode
        finally:
            stop_process(child)
            tee.join(timeout=30)
            child.stdout.close()


def run_worker(args, result_root: Path) -> int:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if (code := read_result(result_root)) is not None:
            return code
        try:
            with socket.create_connection((args.master_addr, RAY_PORT), timeout=2):
                break
        except OSError:
            time.sleep(2)
    else:
        raise TimeoutError("Ray head did not start")
    subprocess.run(ray_argv(args, head=False), check=True, timeout=120)
    deadline, misses = time.monotonic() + args.timeout_s, 0
    while time.monotonic() < deadline:
        if (code := read_result(result_root)) is not None:
            return code
        try:
            status = subprocess.run(
                ["ray", "status", f"--address={args.master_addr}:{RAY_PORT}"], capture_output=True, timeout=15
            )
            misses = 0 if status.returncode == 0 else misses + 1
        except subprocess.TimeoutExpired:
            misses += 1
        if misses >= 6:
            raise RuntimeError("Ray head disappeared without a result")
        time.sleep(POLL_S)
    raise TimeoutError("AIR worker exceeded its runtime budget")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--launcher-path", required=True, help="factory(node_args, launcher_argv, resume_root=...) -> Launch"
    )
    parser.add_argument("--input-manifest")
    parser.add_argument("--reference-root")
    parser.add_argument("--launcher-args", default="", help="shell-quoted recipe arguments")
    parser.add_argument("--local-root", type=Path, default=Path("/tmp/miles-air"))
    parser.add_argument("--timeout-s", type=float, default=48 * 3600)
    raw = list(sys.argv[1:] if argv is None else argv)
    split = raw.index("--") if "--" in raw else len(raw)
    args = parser.parse_args(raw[:split])
    args.num_nodes, args.num_gpus_per_node, args.node_rank = (
        int(os.environ[name]) for name in ("NUM_NODES", "LOCAL_WORLD_SIZE", "NODE_RANK")
    )
    args.master_addr, args.attempt_id = os.environ["MASTER_ADDR"], os.environ["MLFLOW_RUN_ID"]
    if (
        not args.volume_root.is_absolute()
        or args.local_root.is_relative_to("/Volumes")
        or not args.local_root.is_absolute()
    ):
        raise ValueError("AIR needs an absolute Volume root and a local scratch root")
    if not 0 <= args.node_rank < args.num_nodes or args.num_gpus_per_node <= 0:
        raise ValueError("invalid AIR node allocation")
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", v) or v in {".", ".."} for v in (args.run_id, args.attempt_id)
    ):
        raise ValueError("run and attempt IDs must be safe path components")
    return args, [*shlex.split(args.launcher_args), *raw[split + 1 :]]


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args, launcher_args = parse_args(argv)
    result_root = args.volume_root / "runs" / args.run_id / "attempts" / args.attempt_id
    signal.signal(signal.SIGTERM, signal_exit)
    signal.signal(signal.SIGINT, signal_exit)
    code, error = 1, None
    try:
        code = run_head(args, launcher_args, result_root) if args.node_rank == 0 else run_worker(args, result_root)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        logger.exception("AIR node %d failed", args.node_rank)
    finally:
        try:
            if ray.is_initialized():
                ray.shutdown()
            subprocess.run(["ray", "stop", "--force"], timeout=60, check=False)
        except Exception as exc:
            logger.exception("Ray cleanup failed")
            code = code or 1
            error = error or f"{type(exc).__name__}: {exc}"
        if args.node_rank == 0:
            fuse.write_bytes(result_root / "result.json", stage_models.json_bytes({"exit_code": code, "error": error}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
