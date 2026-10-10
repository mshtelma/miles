"""Bounded, verified transfers through the UC Volume FUSE mount."""

import errno
import fcntl
import hashlib
import logging
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024
ATTEMPTS = 6
RETRY_BASE_S = 2.0
logger = logging.getLogger(__name__)


def retry(action, *, path: Path):
    for attempt in range(ATTEMPTS):
        try:
            return action()
        except OSError as error:
            if error.errno not in {errno.EIO, errno.ESTALE, errno.ETIMEDOUT, errno.EAGAIN} or attempt == ATTEMPTS - 1:
                raise
            logger.warning("FUSE retry %d for %s: %s", attempt + 1, path, error)
            time.sleep(min(30, RETRY_BASE_S * 2**attempt))


def exists(path: Path) -> bool:
    def inspect():
        try:
            path.stat()
            return True
        except FileNotFoundError:
            return False

    return retry(inspect, path=path)


def read_bounded(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    def read():
        data = bytearray()
        with path.open("rb") as stream:
            while chunk := stream.read(min(CHUNK_BYTES, max_bytes + 1 - len(data))):
                data.extend(chunk)
                if len(data) > max_bytes:
                    raise ValueError(f"document exceeds {max_bytes} bytes: {path}")
        return bytes(data)

    return retry(read, path=path)


def read_span(path: Path, offset: int, length: int) -> bytes:
    def read():
        data = bytearray()
        with path.open("rb") as stream:
            stream.seek(offset)
            while len(data) < length:
                chunk = stream.read(min(CHUNK_BYTES, length - len(data)))
                if not chunk:
                    raise ValueError(f"short checkpoint storage span: {path}")
                data.extend(chunk)
        return bytes(data)

    return retry(read, path=path)


def digest(path: Path) -> tuple[int, str]:
    checksum, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            checksum.update(chunk)
            size += len(chunk)
    return size, checksum.hexdigest()


def verify(path: Path, *, size_bytes: int, sha256: str) -> None:
    if retry(lambda: digest(path), path=path) != (size_bytes, sha256):
        raise ValueError(f"size/SHA-256 mismatch: {path}")


class NodeTransferLimiter:
    def __init__(self, slot_root: Path, workers: int, *, wait_timeout_s: float = 3600):
        if workers <= 0:
            raise ValueError("transfer workers must be positive")
        self.slot_root, self.workers, self.wait_timeout_s = slot_root, workers, wait_timeout_s
        slot_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def slot(self):
        deadline = time.monotonic() + self.wait_timeout_s
        while True:
            for index in range(self.workers):
                stream = (self.slot_root / str(index)).open("a+b")
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    stream.close()
                    continue
                try:
                    yield
                finally:
                    stream.close()
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"transfer slots stayed busy: {self.slot_root}")
            time.sleep(0.05)


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _stream_copy(source: Path, destination: Path, *, size_bytes: int, sha256: str) -> None:
    checksum, size = hashlib.sha256(), 0
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while chunk := incoming.read(CHUNK_BYTES):
            outgoing.write(chunk)
            checksum.update(chunk)
            size += len(chunk)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if (size, checksum.hexdigest()) != (size_bytes, sha256):
        raise ValueError(f"source size/SHA-256 mismatch: {source}")


def upload_verified(source: Path, destination: Path, *, size_bytes: int, sha256: str, repair: bool = False) -> None:
    if exists(destination):
        try:
            verify(destination, size_bytes=size_bytes, sha256=sha256)
            return
        except ValueError:
            if not repair:
                raise
            retry(destination.unlink, path=destination)

    def upload():
        retry(lambda: destination.parent.mkdir(parents=True, exist_ok=True), path=destination.parent)
        try:
            _stream_copy(source, destination, size_bytes=size_bytes, sha256=sha256)
        except BaseException:
            retry(lambda: destination.unlink(missing_ok=True), path=destination)
            raise

    retry(upload, path=destination)
    # A writer can see EIO when it reopens new Volume bytes. Retry the read-back,
    # without re-uploading a file that already reached close/fsync.
    verify(destination, size_bytes=size_bytes, sha256=sha256)


def download_verified(source: Path, destination: Path, *, size_bytes: int, sha256: str) -> None:
    if destination.exists():
        verify(destination, size_bytes=size_bytes, sha256=sha256)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)

    def download():
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        temporary.unlink()
        try:
            _stream_copy(source, temporary, size_bytes=size_bytes, sha256=sha256)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    retry(download, path=source)


def write_bytes(path: Path, data: bytes, *, overwrite: bool = False, repair: bool = False) -> None:
    with tempfile.NamedTemporaryFile(prefix="miles-volume-") as stream:
        stream.write(data)
        stream.flush()
        if overwrite:
            retry(lambda: path.unlink(missing_ok=True), path=path)
        upload_verified(
            Path(stream.name), path, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), repair=repair
        )
