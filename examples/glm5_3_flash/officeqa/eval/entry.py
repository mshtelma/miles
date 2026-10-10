"""Evaluate pinned HF weights on OfficeQA with one standalone SGLang server per AIR node."""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import time
from pathlib import Path

from examples.glm5_3_flash.officeqa import settings
from examples.glm5_3_flash.officeqa.eval import driver, proxy, server
from tools.air import fuse_transfer as fuse
from tools.air import process as process_utils
from tools.air import stage_models

POLL_S = 5
logger = logging.getLogger(__name__)


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-manifest")
    parser.add_argument("--local-root", type=Path, default=Path("/tmp/miles-air"))
    parser.add_argument("--hf-source", required=True)
    parser.add_argument("--eval-set", action="append", metavar="NAME=PATH")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=480)
    parser.add_argument("--max-seq-len", type=int, default=150_000)
    parser.add_argument("--max-lost-fraction", type=float, default=0.05)
    parser.add_argument("--capture-trajectories", type=int, default=4)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=7 * 3600)
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    args = parser.parse_args(argv)
    args.rank, args.num_nodes = int(os.environ["NODE_RANK"]), int(os.environ["NUM_NODES"])
    if int(os.environ["LOCAL_WORLD_SIZE"]) != 8:
        raise ValueError("the OfficeQA eval engine uses TP8/EP8")
    if (
        min(args.samples_per_prompt, args.max_concurrent, args.max_seq_len, args.timeout_s) <= 0
        or not 0 <= args.max_lost_fraction <= 1
    ):
        raise ValueError("eval counts/deadline must be positive and the lost fraction in [0, 1]")
    args.attempt_id = os.environ["MLFLOW_RUN_ID"]
    args.root = args.volume_root / "runs" / args.run_id / "attempts" / args.attempt_id
    args.manifest = process_utils.resolve_path(args.input_manifest, args.volume_root / "inputs/latest")
    args.input_root = args.local_root / "inputs" / args.manifest.stem
    args.state_root = args.local_root / "runs" / args.run_id / args.attempt_id
    args.eval_sets = _eval_sets(args.eval_set or ["heldout=data/officeqa/eval_heldout.jsonl"], args.input_root)
    return args


def _eval_sets(values: list[str], input_root: Path) -> dict[str, Path]:
    sets = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path or name in sets:
            raise ValueError(f"--eval-set needs a unique NAME=PATH: {value!r}")
        sets[name] = Path(path) if Path(path).is_absolute() else input_root / path
    return sets


def _poll_json(path: Path):
    if not fuse.exists(path):
        return None
    try:
        return stage_models.read_json(path)
    except json.JSONDecodeError:
        return None


def _failures(root: Path) -> list[dict]:
    paths = fuse.retry(lambda: list((root / "failures").glob("node_*.json")), path=root)
    return [value for path in paths if (value := _poll_json(path)) is not None]


def _write_failure(root: Path, rank: int, error: BaseException):
    detail = {"node_rank": rank, "cause_type": type(error).__name__, "cause": str(error)}
    logger.error("OfficeQA eval node failed: %s", detail)
    fuse.write_bytes(root / "failures" / f"node_{rank}.json", stage_models.json_bytes(detail))


def _wait_for_servers(root: Path, num_nodes: int, identity: dict, *, deadline: float, process=None) -> list[str]:
    urls = {}
    while len(urls) < num_nodes:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"head eval engine exited while waiting for servers: {process.returncode}")
        if failures := _failures(root):
            raise RuntimeError(f"eval nodes failed before serving: {failures}")
        for rank in range(num_nodes):
            if rank in urls:
                continue
            value = _poll_json(root / "servers" / f"node_{rank}.json")
            if value is not None:
                if value["model"] != identity:
                    raise ValueError("eval servers staged different models")
                urls[rank] = value["url"]
        if time.monotonic() >= deadline:
            raise TimeoutError(f"only {len(urls)} of {num_nodes} eval servers registered")
        if len(urls) < num_nodes:
            time.sleep(POLL_S)
    return [urls[rank] for rank in range(num_nodes)]


def _serve_until_done(root: Path, process, *, deadline: float) -> int:
    while time.monotonic() < deadline:
        if (code := process_utils.read_result(root)) is not None:
            return code
        if process.poll() is not None:
            raise RuntimeError(f"SGLang exited with code {process.returncode} during eval")
        if failures := _failures(root):
            raise RuntimeError(f"another eval node failed: {failures}")
        time.sleep(POLL_S)
    raise TimeoutError("eval exceeded its runtime budget")


def _routable_ip(master_addr: str) -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((master_addr, 9))
        return probe.getsockname()[0]


def _start_server(args, log_path: Path, deadline: float):
    started = time.monotonic()
    limiter = fuse.NodeTransferLimiter(args.state_root / "transfer-slots", max(1, 32 // args.num_nodes))
    staged = stage_models.stage_hf(
        args.hf_source, args.manifest, args.input_root, "GLM-5.3-Flash", limiter, datasets=args.rank == 0
    )
    stage_s = time.monotonic() - started
    host = _routable_ip(os.environ["MASTER_ADDR"])
    process = server.launch(
        server.server_argv(staged.path, host=host, port=args.port, seed=1234 + args.rank), log_path=log_path
    )
    try:
        url = f"http://{host}:{args.port}"
        ready_s = server.wait_ready(url, process, timeout_s=min(1800, deadline - time.monotonic()))
        mismatches = server.parity_mismatches(server.read_server_args(log_path.read_text(errors="replace")))
        if mismatches:
            raise RuntimeError("SGLang settings differ from training: " + "; ".join(mismatches))
        fuse.write_bytes(
            args.root / "servers" / f"node_{args.rank}.json",
            stage_models.json_bytes({"url": url, "model": staged.identity, "stage_s": stage_s, "ready_s": ready_s}),
        )
    except BaseException:
        process_utils.stop_process(process)
        raise
    return staged, process, {"stage_s": stage_s, "ready_s": ready_s}


def _token_counter(model_path: Path):
    # Only the head renders prompts; tools must be normalized as in training.
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.template import apply_chat_template

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    def count(messages, tools):
        return len(
            apply_chat_template(
                messages,
                tokenizer=tokenizer,
                tools=tools,
                add_generation_prompt=True,
                tokenize=True,
                **server.TEMPLATE_KWARGS,
            )
        )

    return count


async def _evaluate(args, urls, model_path, process, deadline):
    chat = proxy.ChatProxy(
        urls,
        count_prompt_tokens=_token_counter(model_path),
        max_seq_len=args.max_seq_len,
        capture_trajectories=args.capture_trajectories,
    )
    runner, url = await proxy.serve(chat)
    task = None
    try:
        # The agent reads the environment set by main.
        from examples.glm5_3_flash.agent import run as agent_run

        task = asyncio.create_task(
            driver.run_sets(
                {
                    name: driver.load_rows(path)[: args.limit_questions or None]
                    for name, path in args.eval_sets.items()
                },
                agent_run=agent_run,
                proxy=chat,
                proxy_url=url,
                samples_per_prompt=args.samples_per_prompt,
                max_concurrent=args.max_concurrent,
                max_seq_len=args.max_seq_len,
                attempt_id=args.attempt_id,
            )
        )
        while not task.done():
            if process.poll() is not None:
                raise RuntimeError(f"head eval engine exited during scoring: {process.returncode}")
            if failures := await asyncio.to_thread(_failures, args.root):
                raise RuntimeError(f"eval engines failed during scoring: {failures}")
            if time.monotonic() >= deadline:
                raise TimeoutError("eval exceeded its runtime budget")
            await asyncio.wait([task], timeout=POLL_S)
        return await task, {**chat.summary(), "captured": chat.captured}
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()


def _run_head(args, staged, process, ready_s, deadline) -> int:
    urls = _wait_for_servers(args.root, args.num_nodes, staged.identity, deadline=deadline, process=process)
    started = time.monotonic()
    outcomes, harness = asyncio.run(_evaluate(args, urls, staged.path, process, deadline))
    lost = driver.lost_fraction(outcomes)
    valid = lost <= args.max_lost_fraction
    step = staged.identity.get("rollout_id")
    metrics = {
        key: value for name, outcome in outcomes.items() for key, value in driver.set_metrics(name, outcome).items()
    }
    metrics.update(
        {
            f"officeqa_eval/harness/{key}": value
            for key, value in {
                **harness,
                **ready_s,
                "engines": len(urls),
                "eval_s": time.monotonic() - started,
                "lost_fraction": lost,
            }.items()
            if key != "captured"
        }
    )
    results = driver.results_document(
        run_id=args.run_id, rollout_id=step, identity=staged.identity, outcomes=outcomes, valid=valid
    )
    driver.write_outputs(
        args.root / "eval",
        results=results,
        summary={"valid": valid, "metrics": metrics, "model": staged.identity},
        outcomes=outcomes,
        captured=harness["captured"],
        write_once=fuse.write_bytes,
    )
    _log_mlflow(args, metrics, staged.identity, step or 0)
    print(
        "OfficeQA eval: " + json.dumps({"valid": valid, "lost_fraction": lost, "model": staged.identity}), flush=True
    )
    return 0 if valid else 1


def _log_mlflow(args, metrics, identity, step):
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
        with mlflow.start_run(run_id=args.attempt_id):
            mlflow.log_params({"hf_source": args.hf_source, "model": json.dumps(identity, sort_keys=True)[:500]})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)
    except Exception:
        logger.exception("MLflow logging failed; eval results are on the Volume")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse(argv)
    signal.signal(signal.SIGTERM, process_utils.signal_exit)
    signal.signal(signal.SIGINT, process_utils.signal_exit)
    os.environ.update(settings.agent_env(str(args.input_root / "data")))
    process, code = None, 1
    deadline = time.monotonic() + args.timeout_s
    log_path = args.state_root / f"sglang_node_{args.rank}.log"
    with process_utils.LogPublisher(log_path, args.root / "logs" / f"server_node_{args.rank}"):
        try:
            staged, process, ready_s = _start_server(args, log_path, deadline)
            code = (
                _run_head(args, staged, process, ready_s, deadline)
                if args.rank == 0
                else _serve_until_done(args.root, process, deadline=deadline)
            )
        except BaseException as error:
            _write_failure(args.root, args.rank, error)
        finally:
            process_utils.stop_process(process)
            if args.rank == 0:
                fuse.write_bytes(args.root / "result.json", stage_models.json_bytes({"exit_code": code}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
