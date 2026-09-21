import json

from miles.backends.training_utils.weight_update.snapshot_publisher import SnapshotPublisher


def test_snapshot_publisher_commits_writer_output(tmp_path):
    destination = tmp_path / "snapshot"
    calls = []

    def writer(staging, store):
        calls.append(staging)
        store.atomic_write_bytes(staging / "payload", b"snapshot")

    SnapshotPublisher().publish(destination, writer, metadata={"step": 3})

    assert (destination / "payload").read_bytes() == b"snapshot"
    assert json.loads((destination / "META.json").read_text()) == {"step": 3}
    assert len(calls) == 1
