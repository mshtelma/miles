"""Select complete saves and reject incompatible training configurations before launch."""

import hashlib
import json
import logging
from pathlib import Path

from tools.air import dcp_stage
from tools.air import fuse_transfer as fuse
from tools.air.stage_models import json_bytes, model_record, read_complete, safe_path, validate_files

logger = logging.getLogger(__name__)
_LAYOUT = "tensor-model-parallel-size context-parallel-size pipeline-model-parallel-size expert-model-parallel-size expert-tensor-parallel-size virtual-pipeline-model-parallel-size decoder-first-pipeline-num-layers decoder-last-pipeline-num-layers actor-num-nodes actor-num-gpus-per-node sequence-parallel".split()
_OPTIMIZER = "optimizer lr min-lr weight-decay start-weight-decay end-weight-decay weight-decay-incr-style adam-beta1 adam-beta2 adam-eps clip-grad use-distributed-optimizer use-precision-aware-optimizer main-grads-dtype main-params-dtype exp-avg-dtype exp-avg-sq-dtype bf16 fp16".split()
_SCHEDULE = "lr-decay-style lr-decay-iters lr-warmup-iters lr-warmup-fraction lr-warmup-init lr-wsd-decay-iters lr-wsd-decay-style".split()
_MODEL = "spec num-layers hidden-size ffn-hidden-size num-attention-heads num-experts moe-layer-freq q-lora-rank kv-lora-rank padded-vocab-size".split()


def options(argv: list[str]) -> dict[str, list[str]]:
    values, flag = {}, None
    for token in argv:
        if token.startswith("--"):
            flag, separator, value = token[2:].partition("=")
            values[flag] = [value] if separator else []
        elif flag is not None:
            values[flag].append(token)
    return values


def _renderer(flags: dict, model: dict) -> dict:
    # Resolve the same generic template registration as the training parser.
    from miles.utils.chat_template_utils import resolve_fixed_chat_template

    family = flags.get("tito-model", ["default"])[0]
    fixed_path, fixed_kwargs = resolve_fixed_chat_template(family)
    explicit = flags.get("chat-template-path", [None])[0]
    if family != "default" and explicit is not None:
        raise ValueError("a named --tito-model owns its chat template")
    template = fixed_path or explicit
    digest = None
    if template and template != "autofix":
        path = Path(template)
        hf_root = Path(flags["hf-checkpoint"][0])
        if path.is_relative_to(hf_root):
            digest = validate_files(model["files"])[path.relative_to(hf_root).as_posix()]["sha256"]
        else:
            digest = hashlib.sha256(fuse.read_bounded(path)).hexdigest()
    kwargs = json.loads(flags.get("apply-chat-template-kwargs", ["{}"])[0])
    return {
        "family": family,
        "template_sha256": digest,
        "kwargs": {**kwargs, **fixed_kwargs},
        "apply_chat_template": "apply-chat-template" in flags,
        "use_session_server": "use-session-server" in flags,
    }


def configuration(argv: list[str], inputs: dict, model_name: str, input_digest: str, reference: dict) -> dict:
    flags = options(argv)
    forbidden = {
        "async-save",
        "no-save-optim",
        "no-save-rng",
        "no-load-optim",
        "no-load-rng",
        "finetune",
        "reset-rollout-data-source",
        "debug-disable-optimizer",
        "ckpt-step",
        "start-rollout-id",
        "override-opt-param-scheduler",
        "override-ckpt-iteration",
    } & flags.keys()
    if flags.get("advantage-estimator") == ["ppo"]:
        forbidden.add("advantage-estimator=ppo (separate critic checkpoint)")
    if forbidden or flags.get("ckpt-format") != ["torch_dist"]:
        raise ValueError(
            f"AIR needs synchronous full-state torch_dist saves and cursor restoration; incompatible flags: {sorted(forbidden)}"
        )
    model = model_record(inputs, model_name)

    def fields(names):
        return {name: flags.get(name) for name in names}

    def integer(name):
        return int(flags[name][0])

    rollouts, batch, samples, global_batch = (
        integer(name) for name in ("num-rollout", "rollout-batch-size", "n-samples-per-prompt", "global-batch-size")
    )
    train_iters = rollouts * batch * samples // global_batch
    decay_iters = int(flags.get("lr-decay-iters", [train_iters])[0])
    tokenizer = [
        f
        for f in model["files"]
        if any(
            word in f["path"]
            for word in ("tokenizer", "vocab", "merges", "special_tokens", "added_tokens", "chat_template")
        )
    ]
    return {
        "model": {"repo": model["repo"], "revision": model["revision"], "architecture": fields(_MODEL)},
        "tokenizer": {
            "revision": model["revision"],
            "files": tokenizer,
            "renderer": _renderer(flags, model),
        },
        "parallel_layout": fields(_LAYOUT),
        "optimizer": fields(_OPTIMIZER),
        "lr_schedule": {
            **fields(_SCHEDULE),
            "num_rollout": rollouts,
            "global_batch_size": global_batch,
            "rollout_batch_size": batch,
            "n_samples_per_prompt": samples,
            "lr_decay_steps": decay_iters * global_batch,
            "wd_incr_steps": train_iters * global_batch,
        },
        "inputs": {
            "manifest_sha256": input_digest,
            "reference_sha256": hashlib.sha256(json_bytes(reference)).hexdigest(),
        },
    }


def fingerprints(config: dict) -> dict[str, str]:
    return {name: hashlib.sha256(json_bytes(value)).hexdigest() for name, value in config.items()}


def _horizon_error(old: dict, new: dict) -> ValueError:
    return ValueError(
        f"resume changes --num-rollout / LR schedule horizon: saved num_rollout={old['num_rollout']}, lr_decay_steps={old['lr_decay_steps']}; requested num_rollout={new['num_rollout']}, lr_decay_steps={new['lr_decay_steps']}. Keep the original horizon or use a new RUN_ID."
    )


def check_compatibility(saved: dict, requested: dict) -> None:
    previous = saved["configuration"]
    changed = [name for name in requested if previous.get(name) != requested[name]]
    if "lr_schedule" in changed:
        old, new = previous["lr_schedule"], requested["lr_schedule"]
        raise _horizon_error(old, new)
    if changed:
        raise ValueError(f"resume compatibility mismatch: {', '.join(changed)}")


def saved_step(document: dict) -> int:
    step = document.get("rollout_id")
    if type(step) is not int or step < 0:
        raise ValueError("save has an invalid rollout ID")
    return step


def _validate_save(root: Path, document: dict, step: int) -> None:
    entries = validate_files(document["files"])
    required = {
        "latest_checkpointed_iteration.txt",
        f"iter_{step:07d}/.metadata",
        f"iter_{step:07d}/metadata.json",
        f"rollout/global_dataset_state_dict_{step}.pt",
    }
    if not required <= entries.keys():
        raise ValueError("save lacks its tracker, DCP metadata or rollout cursor")
    tracker = fuse.read_bounded(root / "latest_checkpointed_iteration.txt", max_bytes=1024)
    if tracker.strip() != str(step).encode():
        raise ValueError("save tracker differs from its rollout ID")
    if entries["latest_checkpointed_iteration.txt"]["sha256"] != hashlib.sha256(tracker).hexdigest():
        raise ValueError("save tracker differs from its manifest")
    for name, entry in entries.items():
        path = safe_path(root, name)
        if fuse.retry(lambda p=path: p.stat().st_size, path=path) != entry["size_bytes"]:
            raise ValueError(f"save file is missing or short: {path}")
    dcp_stage.validate_checkpoint(root / f"iter_{step:07d}", document["files"], step)


def latest_complete(volume_root: Path, run_id: str) -> tuple[Path, dict] | None:
    root = volume_root / "runs" / run_id / "checkpoints"
    markers = fuse.retry(lambda: list(root.glob("*/*/COMPLETE.json")), path=root)
    candidates = []
    for marker in markers:
        try:
            document = read_complete(marker.parent)
            step = saved_step(document)
            if document.get("run_id") != run_id or document.get("attempt_id") != marker.parent.parent.name:
                raise ValueError("save belongs to a different run/attempt")
            _validate_save(marker.parent, document, step)
            candidates.append((step, marker.parent, document))
        except (ValueError, KeyError, FileNotFoundError) as error:
            logger.warning("Skipping incomplete save %s: %s", marker.parent, error)
    if not candidates:
        if markers:
            raise ValueError("run contains COMPLETE markers but no valid save")
        return None
    newest = max(step for step, _, _ in candidates)
    winners = [(path, document) for step, path, document in candidates if step == newest]
    if len(winners) != 1:
        raise ValueError("run has multiple COMPLETE saves at its newest rollout ID")
    return winners[0]
