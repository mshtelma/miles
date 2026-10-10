"""Child process cleanup, periodic log publication and a shared head outcome."""

import json
import logging
import os
import signal
import subprocess
import threading
from pathlib import Path

from tools.air import fuse_transfer as fuse
from tools.air.stage_models import read_json

LOG_CHUNK_BYTES = 4 * 1024 * 1024
logger = logging.getLogger(__name__)


def signal_exit(signum, _frame):
    raise SystemExit(128 + signum)


def stop_process(process) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


class LogPublisher:
    def __init__(self, source: Path, target: Path):
        self.source, self.target = source, target
        self.offset = 0
        self._pending = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def flush(self) -> None:
        if not self.source.exists():
            return
        end = self.source.stat().st_size
        with self.source.open("rb") as stream:
            while self.offset < end:
                if self._pending is None:
                    stream.seek(self.offset)
                    self._pending = stream.read(min(LOG_CHUNK_BYTES, end - self.offset))
                fuse.write_bytes(self.target / f"{self.offset:016d}.log", self._pending, repair=True)
                self.offset += len(self._pending)
                self._pending = None

    def _run(self):
        while True:
            stopping = self._stop.is_set()
            try:
                self.flush()
            except Exception:
                logger.exception("could not publish log chunks from %s", self.source)
            if stopping:
                return
            self._stop.wait(30)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=60)


def resolve_path(value: str | None, pointer: Path) -> Path:
    return Path(value) if value else Path(fuse.read_bounded(pointer, max_bytes=4096).decode().strip())


def read_result(root: Path) -> int | None:
    if not fuse.exists(root / "result.json"):
        return None
    try:
        return int(read_json(root / "result.json")["exit_code"])
    except json.JSONDecodeError:
        return None
