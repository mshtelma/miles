"""Head-node pinning must not depend on a dashboard reachable from the caller."""

import pytest
import ray

from miles.utils import ray_utils


_WORKER, _OLD_HEAD, _HEAD = ("1" * 56, "2" * 56, "3" * 56)  # Ray node IDs are 56 hex characters


def _node(node_id: str, *, alive: bool = True, head: bool = False) -> dict:
    resources = {"CPU": 8.0, f"node:10.0.0.{node_id[0]}": 1.0}
    if head:
        resources["node:__internal_head__"] = 1.0
    return {"NodeID": node_id, "NodeManagerAddress": f"10.0.0.{node_id[0]}", "Alive": alive, "Resources": resources}


def test_head_node_is_found_from_the_gcs_node_table(monkeypatch):
    nodes = [_node(_WORKER), _node(_OLD_HEAD, alive=False, head=True), _node(_HEAD, head=True)]
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: nodes)

    options = ray_utils.compute_ray_pin_head_options()

    assert options["scheduling_strategy"].node_id == _HEAD
    assert options["scheduling_strategy"].soft is False


def test_head_node_ip_is_the_live_head_nodes_address(monkeypatch):
    nodes = [_node(_WORKER), _node(_OLD_HEAD, alive=False, head=True), _node(_HEAD, head=True)]
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: nodes)

    assert ray_utils.get_head_node_ip() == "10.0.0.3"


def test_missing_head_node_is_an_error(monkeypatch):
    monkeypatch.setattr(ray_utils.ray, "nodes", lambda: [_node(_WORKER), _node(_OLD_HEAD, alive=False, head=True)])

    with pytest.raises(RuntimeError, match="head node"):
        ray_utils._get_head_node_id()


def test_live_cluster_without_a_dashboard_reports_its_head(ray_local_mode):
    """The session cluster starts without a dashboard; the driver runs on its head node."""
    assert ray_utils._get_head_node_id() == ray.get_runtime_context().get_node_id()
