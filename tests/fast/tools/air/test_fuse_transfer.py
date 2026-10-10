import errno
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from tools.air import fuse_transfer as fuse
from tools.air import stage_models


def _payload(tmp_path):
    source, target = tmp_path / "source", tmp_path / "volume/file"
    raw = b"checkpoint" * 100
    source.write_bytes(raw)
    return source, target, {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_read_back_eio_retries_without_reupload(tmp_path, monkeypatch):
    source, target, expected = _payload(tmp_path)
    original_digest, original_copy = fuse.digest, fuse._stream_copy
    writes, reads = [], []

    def digest(path):
        if path == target:
            reads.append(path)
            if len(reads) == 1:
                raise OSError(errno.EIO, "new FUSE bytes are not readable yet")
        return original_digest(path)

    def copy(*args, **kwargs):
        writes.append(args[1])
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(fuse, "digest", digest)
    monkeypatch.setattr(fuse, "_stream_copy", copy)
    monkeypatch.setattr(fuse, "RETRY_BASE_S", 0)
    fuse.upload_verified(source, target, **expected)
    assert len(reads) == 2 and writes == [target]
    assert target.read_bytes() == source.read_bytes()


def test_interrupted_upload_is_cleaned_and_a_retry_succeeds(tmp_path, monkeypatch):
    source, target, expected = _payload(tmp_path)
    original = fuse._stream_copy

    def interrupted(_source, destination, **kwargs):
        destination.write_bytes(b"partial")
        raise OSError(errno.EIO, "interrupted upload")

    monkeypatch.setattr(fuse, "ATTEMPTS", 1)
    monkeypatch.setattr(fuse, "_stream_copy", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        fuse.upload_verified(source, target, **expected)
    assert not target.exists()
    monkeypatch.setattr(fuse, "_stream_copy", original)
    fuse.upload_verified(source, target, **expected)
    assert target.read_bytes() == source.read_bytes()


def test_killed_upload_can_repair_partial_destination(tmp_path):
    source, target, expected = _payload(tmp_path)
    target.parent.mkdir()
    target.write_bytes(b"partial")
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        fuse.upload_verified(source, target, **expected)
    fuse.upload_verified(source, target, **expected, repair=True)
    assert target.read_bytes() == source.read_bytes()


def test_failed_download_leaves_no_final_file_or_temporary_file(tmp_path, monkeypatch):
    source, target, expected = _payload(tmp_path)

    def interrupted(_source, destination, **kwargs):
        destination.write_bytes(b"partial")
        raise OSError(errno.EIO, "interrupted download")

    monkeypatch.setattr(fuse, "ATTEMPTS", 1)
    monkeypatch.setattr(fuse, "_stream_copy", interrupted)
    with pytest.raises(OSError):
        fuse.download_verified(source, target, **expected)
    assert list(target.parent.iterdir()) == []


def test_documents_and_spans_use_bounded_reads(tmp_path, monkeypatch):
    path = tmp_path / "document"
    path.write_bytes(b"0123456789abcdef")
    monkeypatch.setattr(fuse, "CHUNK_BYTES", 4)
    assert fuse.read_bounded(path) == path.read_bytes()
    assert fuse.read_span(path, 2, 10) == b"23456789ab"
    with pytest.raises(ValueError, match="document exceeds"):
        fuse.read_bounded(path, max_bytes=8)
    with pytest.raises(ValueError, match="short checkpoint"):
        fuse.read_span(path, 10, 20)


def test_node_limiters_bound_combined_concurrent_transfers(tmp_path):
    limiters = [fuse.NodeTransferLimiter(tmp_path / "slots", 2) for _ in range(2)]
    lock = threading.Lock()
    active = peak = 0

    def work(index):
        nonlocal active, peak
        with limiters[index % 2].slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.01)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(8)))
    assert 1 <= peak <= 2


def test_partial_manifest_and_marker_can_be_retried_but_complete_is_immutable(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "weights").write_bytes(b"weights")
    document = {"schema": 1, "kind": "reference", "files": stage_models.inventory(root)}
    (root / "manifest.json").write_bytes(b"{")
    (root / "COMPLETE.json").write_bytes(b"{")
    stage_models.write_complete(root, document)
    assert stage_models.read_complete(root) == document
    original = (root / "COMPLETE.json").read_bytes()
    stage_models.write_complete(root, document)
    with pytest.raises(ValueError, match="completed artifact differs"):
        stage_models.write_complete(root, {**document, "kind": "other"})
    assert (root / "COMPLETE.json").read_bytes() == original
