"""Adapt the OfficeQA launcher to the shared AIR node supervisor."""

import json
import shlex
import sys
from pathlib import Path

from scripts import run_glm5_3_flash as launcher
from tools.air.node_entry import Launch
from tools.air.resume import options
from typer.main import get_command

from miles.utils.external_utils.model_args_utils import load_model_args


def _without(argv: list[str], flags: set[str]) -> list[str]:
    kept, skip = [], False
    for token in argv:
        if skip:
            skip = False
        elif token.split("=", 1)[0] in flags:
            skip = "=" not in token
        else:
            kept.append(token)
    return kept


def make_launch(node, argv: list[str], *, resume_root: Path | None) -> Launch:
    owned = {
        "run-id": node.run_id,
        "num-nodes": str(node.num_nodes),
        "num-gpus-per-node": str(node.num_gpus_per_node),
        "output-dir": str(node.state_root),
        "model-dir": str(node.input_root / "models"),
        "data-dir": str(node.input_root / "data"),
        "hf-checkpoint": str(node.input_root / "models/GLM-5.3-Flash"),
        "prompt-data": str(node.input_root / "data/officeqa/train.jsonl"),
        "ckpt-dir": str(node.state_root / "dcp/reference"),
    }
    if any(token.split("=", 1)[0].removeprefix("--") in {*owned, "resume-root"} for token in argv):
        raise ValueError("node entry owns the launcher's run, allocation, input and resume paths")
    if resume_root is not None:
        owned["resume-root"] = str(resume_root)
    base = [item for flag, value in owned.items() for item in ("--" + flag, value)]
    command = get_command(launcher.app).commands["train"]
    with command.make_context("train", base + argv) as context:
        args = launcher.ScriptArgs(**context.params)
    reserved = {
        "load",
        "ref-load",
        "save",
        "save-hf",
        "hf-checkpoint",
        "prompt-data",
        "checkpoint-publisher-path",
        "custom-megatron-init-path",
        "custom-megatron-post-save-hook-path",
        "mlflow-run-id",
    } & options(shlex.split(args.extra_args)).keys()
    if reserved:
        raise ValueError(f"node entry owns checkpoint integration flags: {sorted(reserved)}")
    hooks = ["--custom-megatron-init-path", "tools.air.dcp_stage.initialize"]
    if not args.skip_saving:
        hooks += ["--checkpoint-publisher-path", "tools.air.publisher.CheckpointPublisher"]
    if args.mlflow_experiment is not None:
        hooks += ["--mlflow-run-id", node.attempt_id]
    args.extra_args = shlex.join([*shlex.split(args.extra_args), *hooks])
    env = launcher.U.resolve_extra_env_vars({}, args)
    env.update(MILES_CHECKPOINT_CONFIG=str(node.state_root / "runtime.json"), MILES_ATTEMPT_ID=node.attempt_id)
    args.extra_env_vars = json.dumps(env)
    training_args = [*shlex.split(load_model_args(launcher._MODEL_TYPE)), *launcher._train_args(args)]
    forwarded = _without(argv, {"--extra-args", "--extra-env-vars"})
    return Launch(
        command=[
            sys.executable,
            str(launcher.U.repo_base_dir / "scripts/run_glm5_3_flash.py"),
            "train",
            *base,
            *forwarded,
            "--extra-args",
            args.extra_args,
            "--extra-env-vars",
            args.extra_env_vars,
        ],
        training_args=training_args,
        model_name=Path(args.hf_checkpoint).name,
        reference_name=f"{launcher._MODEL_TYPE}_torch_dist",
        save_hf=args.save_hf,
    )
