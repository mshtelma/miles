"""Publish synchronous distributed saves with Ray RPC and write COMPLETE last."""

import asyncio
import hashlib
import os
import shutil
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tools.air import dcp_stage
from tools.air import fuse_transfer as fuse
from tools.air import resume, stage_models


def inventory_node(checkpoint_root: Path, rollout_id: int) -> list[dict]:
    iteration = checkpoint_root / f"iter_{rollout_id:07d}"
    paths = list(iteration.rglob("*")) if iteration.exists() else []
    for path in (
        checkpoint_root / "latest_checkpointed_iteration.txt",
        checkpoint_root / f"rollout/global_dataset_state_dict_{rollout_id}.pt",
    ):
        if path.exists():
            paths.append(path)
    return stage_models.inventory(checkpoint_root, paths)


def merge_inventories(inventories: dict[str, list[dict]], node_ids: list[str]) -> list[dict]:
    if set(inventories) != set(node_ids):
        raise ValueError("checkpoint inventory is missing a node")
    chosen = {}
    for node_id in node_ids:
        for name, entry in stage_models.validate_files(inventories[node_id]).items():
            old = chosen.get(name)
            if old is not None and (old["size_bytes"], old["sha256"]) != (entry["size_bytes"], entry["sha256"]):
                raise ValueError(f"conflicting duplicate checkpoint file: {name}")
            if old is None:
                chosen[name] = {**entry, "owner": node_id}
    return [chosen[name] for name in sorted(chosen)]


def require_save_files(files: list[dict], rollout_id: int) -> None:
    entries = stage_models.validate_files(files)
    required = {
        "latest_checkpointed_iteration.txt",
        f"iter_{rollout_id:07d}/.metadata",
        f"iter_{rollout_id:07d}/metadata.json",
        f"rollout/global_dataset_state_dict_{rollout_id}.pt",
    }
    missing = [name for name in required if not entries.get(name, {}).get("size_bytes")]
    if missing:
        raise ValueError(f"checkpoint is missing required files: {sorted(missing)}")
    tracker = entries["latest_checkpointed_iteration.txt"]
    allowed = {
        (len(raw), hashlib.sha256(raw).hexdigest()) for raw in (str(rollout_id).encode(), f"{rollout_id}\n".encode())
    }
    if (tracker["size_bytes"], tracker["sha256"]) not in allowed:
        raise ValueError("checkpoint tracker differs from the saved rollout ID")


def copy_node(checkpoint_root: Path, target: Path, slot_root: Path, workers: int, *, files: list[dict]) -> list[dict]:
    limiter = fuse.NodeTransferLimiter(slot_root, workers)
    stage_models.copy_files(
        checkpoint_root, target, files, limiter, upload=True, repair=not fuse.exists(target / "COMPLETE.json")
    )
    return files


def finalize_save(
    target: Path,
    files: list[dict],
    *,
    run_id: str,
    attempt_id: str,
    rollout_id: int,
    configuration: dict,
    save_hf=False,
) -> dict:
    require_save_files(files, rollout_id)
    for entry in files:
        path = stage_models.safe_path(target, entry["path"])
        if fuse.retry(lambda p=path: p.stat().st_size, path=path) != entry["size_bytes"]:
            raise ValueError(f"published size differs from the inventory: {path}")
    common_keys = dcp_stage.validate_checkpoint(target / f"iter_{rollout_id:07d}", files, rollout_id)
    receipt_name = f"iter_{rollout_id:07d}/hf_export.json"
    export_status = (
        stage_models.read_json(target / receipt_name).get("status")
        if any(f["path"] == receipt_name for f in files)
        else None
    )
    if save_hf and (
        export_status != "ok" or not any(f["path"].startswith(f"iter_{rollout_id:07d}/hf/") for f in files)
    ):
        raise ValueError("requested HF export is missing or did not succeed")
    document = {
        "schema": stage_models.SCHEMA,
        "kind": "checkpoint",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "rollout_id": rollout_id,
        "configuration": configuration,
        "fingerprints": resume.fingerprints(configuration),
        "files": files,
        "common_keys": common_keys,
        "hf_export_status": export_status,
    }
    stage_models.write_complete(target, document)
    return document


def prune_node(checkpoint_root: Path, completed_steps: list[int]) -> list[int]:
    expired = sorted(completed_steps)[:-2]
    for step in expired:
        iteration = checkpoint_root / f"iter_{step:07d}"
        if iteration.exists():
            shutil.rmtree(iteration)
        (checkpoint_root / f"rollout/global_dataset_state_dict_{step}.pt").unlink(missing_ok=True)
    return expired


def capacity_node(config: dict, completed_steps: list[int]) -> dict:
    state, checkpoint = Path(config["state_root"]), Path(config["checkpoint_root"])
    ranks = [stage_models.read_json(path) for path in (state / "trainer-ranks").glob("*.json")]
    # Seven BF16 weight copies cover weights, FP32 master weights and Adam moments.
    # The rank markers come from actual trainer placement, never AIR's NODE_RANK.
    generation = config["reference_bytes"] * 7 * len(ranks) // ranks[0]["world_size"] if ranks else 0
    if config["save_hf"] and any(rank["rank"] == 0 for rank in ranks):
        generation += config["hf_bytes"]
    measured = max(
        (
            sum(p.stat().st_size for p in (checkpoint / f"iter_{step:07d}").rglob("*") if p.is_file())
            for step in completed_steps[-2:]
        ),
        default=0,
    )
    usage = shutil.disk_usage(state)
    required = (measured * 11 // 10 if measured else generation) + usage.total // 50
    return {"trainer_ranks": len(ranks), "free_bytes": usage.free, "required_bytes": required}


async def on_nodes(function, node_ids: list[str], *args, timeout_s: float = 3600, per_node=None, **kwargs):
    task = ray.remote(num_cpus=0, max_retries=0)(function)
    refs = [
        task.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)).remote(
            *args, **{**kwargs, **(per_node or {}).get(node_id, {})}
        )
        for node_id in node_ids
    ]
    try:
        return await asyncio.wait_for(asyncio.gather(*refs), timeout=timeout_s)
    except BaseException:
        for ref in refs:
            ray.cancel(ref, force=True)
        raise


class CheckpointPublisher:
    def __init__(self, args):
        self.config = stage_models.read_json(Path(os.environ["MILES_CHECKPOINT_CONFIG"]))
        self.node_ids = self.config["node_ids"]
        self.checkpoint_root = Path(self.config["checkpoint_root"])
        self.volume_root = Path(self.config["volume_root"])
        self._completed = []
        self._trainer_ranks = args.actor_num_nodes * args.actor_num_gpus_per_node
        if args.save is None or Path(args.save) != self.checkpoint_root:
            raise ValueError("launcher save directory differs from the node's checkpoint root")
        if args.ckpt_format != "torch_dist" or args.no_save_optim or args.no_save_rng:
            raise ValueError("AIR publication needs full-state torch_dist saves")
        schedule = self.config["configuration"]["lr_schedule"]
        if (args.num_rollout, args.global_batch_size, args.rollout_batch_size, args.n_samples_per_prompt) != (
            schedule["num_rollout"],
            schedule["global_batch_size"],
            schedule["rollout_batch_size"],
            schedule["n_samples_per_prompt"],
        ):
            raise ValueError("parsed training arguments differ from the preflight LR horizon")

    async def before_save(self, rollout_id: int) -> None:
        if self._completed and rollout_id <= self._completed[-1]:
            raise ValueError("checkpoint rollout IDs must increase")
        capacities = await on_nodes(capacity_node, self.node_ids, self.config, self._completed, timeout_s=60)
        if sum(c["trainer_ranks"] for c in capacities) != self._trainer_ranks:
            raise ValueError("checkpoint preflight cannot locate every trainer rank")
        short = [
            (node, c["free_bytes"], c["required_bytes"])
            for node, c in zip(self.node_ids, capacities, strict=True)
            if c["free_bytes"] < c["required_bytes"]
        ]
        if short:
            raise RuntimeError(f"insufficient local checkpoint capacity (node, free, required): {short}")

    async def after_save(self, rollout_id: int) -> None:
        inventories = await on_nodes(inventory_node, self.node_ids, self.checkpoint_root, rollout_id)
        files = merge_inventories(dict(zip(self.node_ids, inventories, strict=True)), self.node_ids)
        require_save_files(files, rollout_id)
        target = (
            self.volume_root
            / "runs"
            / self.config["run_id"]
            / "checkpoints"
            / self.config["attempt_id"]
            / f"iter_{rollout_id:07d}"
        )
        copied = await on_nodes(
            copy_node,
            self.node_ids,
            self.checkpoint_root,
            target,
            Path(self.config["slot_root"]),
            self.config["workers"],
            timeout_s=7200,
            per_node={node: {"files": [f for f in files if f["owner"] == node]} for node in self.node_ids},
        )
        returned = [entry for node_files in copied for entry in node_files]
        if sorted(returned, key=lambda f: f["path"]) != files:
            raise ValueError("node copies did not return the full checkpoint inventory")
        await asyncio.to_thread(
            finalize_save,
            target,
            files,
            run_id=self.config["run_id"],
            attempt_id=self.config["attempt_id"],
            rollout_id=rollout_id,
            configuration=self.config["configuration"],
            save_hf=self.config["save_hf"],
        )
        self._completed.append(rollout_id)
        await on_nodes(prune_node, self.node_ids, self.checkpoint_root, self._completed, timeout_s=300)
        print(f"AIR checkpoint COMPLETE: {target}", flush=True)
