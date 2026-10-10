import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Ray gives only the head node this resource; Ray Train and Ray Data find the head the same way.
_HEAD_NODE_RESOURCE = "node:__internal_head__"


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


def compute_ray_pin_head_options():
    head_node_id = _get_head_node_id()
    return {
        "scheduling_strategy": NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        )
    }


def get_head_node_ip() -> str:
    return _get_head_node()["NodeManagerAddress"]


def _get_head_node_id() -> str:
    return _get_head_node()["NodeID"]


def _get_head_node() -> dict:
    # ray.nodes() reads the GCS, so it works from any node. The state API needs the dashboard,
    # which a head started with --dashboard-host=127.0.0.1 serves only on its own loopback.
    for node in ray.nodes():
        if node["Alive"] and _HEAD_NODE_RESOURCE in node["Resources"]:
            return node
    raise RuntimeError("Could not find a head node in the Ray cluster")
