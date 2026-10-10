"""Copy each training save from the nodes' local disks to the Volume while training goes on.

``scripts/run_glm5_3_flash.py --persist-dir DIR`` sets ``--checkpoint-publisher-path`` to
``VolumePublisher`` and exports ``MILES_PERSIST_DIR=DIR``. Megatron saves to ``--save`` on each node's
local disk. Once the trainer and the rollout state are saved, one task per node copies that node's part of
the save into DIR and deletes it locally. Megatron's tracker, ``latest_checkpointed_iteration.txt``, is
written last, so ``--load DIR`` resumes only from a save that was copied whole.
"""

import asyncio
import os
import shutil
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tools.air.copy_tree import copy_tree

TRACKER = "latest_checkpointed_iteration.txt"


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


class VolumePublisher:
    def __init__(self, args):
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
