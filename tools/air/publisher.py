"""Copy each training save from the nodes' local disks to the Volume while training goes on.

``scripts/run_air_officeqa.py --persist-dir DIR`` sets ``--checkpoint-publisher-path`` to
``VolumePublisher`` and exports ``MILES_PERSIST_DIR=DIR``. Megatron saves to ``--save`` on each node's
local disk. Once the trainer and the rollout state are saved, one task per node copies that node's part of
the save into DIR and deletes it locally. Megatron's tracker, ``latest_checkpointed_iteration.txt``, is
written last, so ``--load DIR`` resumes only from a save that was copied whole. The older saves then lose
their training state, about 4.4 TB each, and keep only their HF export.

When the run logs to MLflow, ``mlflow_runs.json`` in DIR maps each save to the MLflow run that made it,
which an eval of the save logs into, and that run gets a ``miles.save.<step>`` tag with the save's path.
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tools.air.copy_tree import copy_tree

TRACKER = "latest_checkpointed_iteration.txt"
MLFLOW_RUNS = "mlflow_runs.json"


def copy_node_save(local: Path, remote: Path, rollout_id: int) -> None:
    """Copy this node's part of a save, then delete it: its ranks' DCP shards and, where they were
    written, the HF export, the DCP metadata and the rollout state."""
    for relative in (f"iter_{rollout_id:07d}", f"rollout/global_dataset_state_dict_{rollout_id}.pt"):
        path = local / relative
        if path.is_dir():
            copy_tree(path, remote / relative)
            shutil.rmtree(path)
        elif path.is_file():
            (remote / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, remote / relative)
            path.unlink()


def prune_older_saves(remote: Path, rollout_id: int) -> None:
    """Delete everything but the HF export from the saves before ``rollout_id``: only the newest save
    is ever resumed from."""
    for save in remote.glob("iter_*"):
        if int(save.name.removeprefix("iter_")) < rollout_id:
            for path in save.iterdir():
                if path.is_dir() and path.name != "hf":
                    shutil.rmtree(path)
                elif path.is_file():
                    path.unlink()
            if not any(save.iterdir()):
                save.rmdir()  # a save without an HF export leaves nothing to keep


def record_mlflow_run(remote: Path, rollout_id: int, run_id: str) -> None:
    """Map the save to its MLflow run, and tag the run with the save's path."""
    # Imported here: MLflow is needed only by runs that log to it.
    from mlflow.tracking import MlflowClient

    runs_file = remote / MLFLOW_RUNS
    runs = json.loads(runs_file.read_text()) if runs_file.is_file() else {}
    runs_file.write_text(json.dumps({**runs, str(rollout_id): run_id}, indent=1, sort_keys=True))
    MlflowClient().set_tag(run_id, f"miles.save.{rollout_id}", str(remote / f"iter_{rollout_id:07d}"))


class VolumePublisher:
    def __init__(self, args):
        # Miles sets args.mlflow_run_id when tracking starts, after the publisher is made.
        self.args = args
        self.local, self.remote = Path(args.save), Path(os.environ["MILES_PERSIST_DIR"])

    async def after_save(self, rollout_id: int) -> None:
        copy = ray.remote(num_cpus=0, max_retries=0)(copy_node_save)
        nodes = [node["NodeID"] for node in ray.nodes() if node["Alive"]]
        await asyncio.gather(
            *(
                copy.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote(
                    self.local, self.remote, rollout_id
                )
                for node in nodes
            )
        )
        await asyncio.to_thread((self.remote / TRACKER).write_text, str(rollout_id))
        print(f"checkpoint {rollout_id} copied to {self.remote}", flush=True)
        if getattr(self.args, "mlflow_run_id", None) is not None:
            try:
                await asyncio.to_thread(record_mlflow_run, self.remote, rollout_id, self.args.mlflow_run_id)
            except Exception as error:  # bookkeeping: a failed MLflow call must not stop the training run
                print(f"could not record checkpoint {rollout_id} in MLflow: {error!r}", flush=True)
        await asyncio.to_thread(prune_older_saves, self.remote, rollout_id)
