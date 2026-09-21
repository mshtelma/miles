import pytest

from miles.utils import distributed_phase
from miles.utils.distributed_phase import DistributedPhaseError, DistributedPhaseGate


def test_local_phase_preserves_error_without_distributed_runtime(monkeypatch):
    error = OSError("disk full")
    monkeypatch.setattr(distributed_phase.dist, "is_initialized", lambda: False)

    def fail():
        raise error

    with pytest.raises(OSError) as caught:
        DistributedPhaseGate().run_local_phase("checkpoint.write", fail)
    assert caught.value is error


def test_local_phase_reports_remote_failures(monkeypatch):
    monkeypatch.setattr(distributed_phase.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(distributed_phase.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(distributed_phase, "get_gloo_group", lambda: object())

    def all_gather_object(output, local_message, group):
        output[:] = [None, "OSError: shard serialization failed"]

    monkeypatch.setattr(distributed_phase.dist, "all_gather_object", all_gather_object)

    with pytest.raises(DistributedPhaseError, match="rank 1: OSError: shard serialization failed"):
        DistributedPhaseGate().run_local_phase("checkpoint.write", lambda: None)
