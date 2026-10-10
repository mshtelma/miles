"""Evaluate an HF checkpoint on OfficeQA with one standalone SGLang server on one AIR node.

The checkpoint is the base model or the HF export inside a training save that has been copied to the Volume
in full: Megatron's tracker there has reached the save, and Miles marked its export complete.
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

from examples.glm5_3_flash.officeqa import settings
from examples.glm5_3_flash.officeqa.eval import driver, proxy, server
from tools.air.copy_tree import copy_tree

logger = logging.getLogger(__name__)


def _parse(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-root", type=Path, default=Path("/tmp/miles-air"))
    parser.add_argument("--hf-source", default="base", help="base, or a save's directory .../checkpoints/iter_<step>")
    parser.add_argument("--eval-set", action="append", metavar="NAME=PATH")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--max-concurrent", type=int, default=480)
    parser.add_argument("--max-seq-len", type=int, default=150_000)
    parser.add_argument("--max-lost-fraction", type=float, default=0.05)
    parser.add_argument("--limit-questions", type=int, default=0)
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    args = parser.parse_args(argv)
    args.attempt_id = os.environ["MLFLOW_RUN_ID"]
    args.out_dir = args.volume_root / "runs" / args.run_id / args.attempt_id
    args.eval_sets = _eval_sets(args.eval_set or ["heldout=data/officeqa/eval_heldout.jsonl"], args.local_root)
    return args


def model_source(volume_root: Path, hf_source: str) -> tuple[Path, dict]:
    """The HF directory to serve, and what the outputs record about it."""
    if hf_source == "base":
        return volume_root / "models/GLM-5.3-Flash", {"kind": "base"}
    save = Path(hf_source)
    step = int(save.name.removeprefix("iter_"))
    copied = int((save.parent / "latest_checkpointed_iteration.txt").read_text())
    if copied < step or not (save / "hf/.complete").is_file():
        raise ValueError(f"{save} is not a fully copied save with a complete HF export")
    return save / "hf", {"kind": "checkpoint", "path": str(save), "rollout_id": step}


def _eval_sets(values: list[str], input_root: Path) -> dict[str, Path]:
    sets = {}
    for value in values:
        name, _, path = value.partition("=")
        if not name or not path or name in sets:
            raise ValueError(f"--eval-set needs a unique NAME=PATH: {value!r}")
        sets[name] = Path(path) if Path(path).is_absolute() else input_root / path
    return sets


def _token_counter(model_path: Path):
    # Tools must be normalized as in training.
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


async def _evaluate(args, url: str, model_path: Path, process):
    chat = proxy.ChatProxy(url, count_prompt_tokens=_token_counter(model_path), max_seq_len=args.max_seq_len)
    runner, proxy_url = await proxy.serve(chat)
    task = None
    try:
        # The agent reads the environment set by main.
        from examples.glm5_3_flash.agent import run as agent_run

        sets = {name: driver.load_rows(path)[: args.limit_questions or None] for name, path in args.eval_sets.items()}
        task = asyncio.create_task(
            driver.run_sets(
                sets,
                agent_run=agent_run,
                proxy=chat,
                proxy_url=proxy_url,
                samples_per_prompt=args.samples_per_prompt,
                max_concurrent=args.max_concurrent,
                max_seq_len=args.max_seq_len,
                attempt_id=args.attempt_id,
            )
        )
        while not task.done():
            if process.poll() is not None:
                raise RuntimeError(f"SGLang exited with code {process.returncode} during the eval")
            await asyncio.wait([task], timeout=5)
        return await task, chat.summary()
    finally:
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await runner.cleanup()


def _log_mlflow(args, metrics, model, step):
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
        with mlflow.start_run(run_id=args.attempt_id):
            mlflow.log_params({"hf_source": args.hf_source, "model": json.dumps(model, sort_keys=True)[:500]})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()}, step=step)
    except Exception:
        logger.exception("MLflow logging failed; eval results are on the Volume")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO)
    args = _parse(argv)
    started = time.monotonic()
    source, model = model_source(args.volume_root, args.hf_source)
    model_path = args.local_root / "model"
    copy_tree(source, model_path)
    copy_tree(args.volume_root / "data/officeqa", args.local_root / "data/officeqa")
    os.environ.update(settings.agent_env(str(args.local_root / "data")))
    timings = {"stage_s": time.monotonic() - started}
    url = f"http://127.0.0.1:{args.port}"
    log_path = args.local_root / "sglang.log"
    process = server.launch(
        server.server_argv(model_path, host="127.0.0.1", port=args.port, seed=1234), log_path=log_path
    )
    try:
        timings["ready_s"] = server.wait_ready(url, process, timeout_s=1800)
        started = time.monotonic()
        outcomes, harness = asyncio.run(_evaluate(args, url, model_path, process))
        timings["eval_s"] = time.monotonic() - started
    finally:
        server.stop(process)
        if log_path.exists():
            args.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(log_path, args.out_dir / "sglang.log")
    lost = driver.lost_fraction(outcomes)
    valid = lost <= args.max_lost_fraction
    metrics = {
        key: value for name, outcome in outcomes.items() for key, value in driver.set_metrics(name, outcome).items()
    }
    metrics.update(
        {f"officeqa_eval/harness/{key}": value for key, value in {**harness, **timings, "lost_fraction": lost}.items()}
    )
    summary = {"valid": valid, "metrics": metrics, "model": model}
    driver.write_outputs(args.out_dir, model=model, summary=summary, outcomes=outcomes)
    _log_mlflow(args, metrics, model, model.get("rollout_id") or 0)
    print("OfficeQA eval: " + json.dumps({"valid": valid, "lost_fraction": lost, "model": model}), flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
