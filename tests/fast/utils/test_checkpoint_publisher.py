import asyncio
from types import SimpleNamespace

import pytest

from miles.utils.checkpoint_publisher import CheckpointPublisher
from miles.utils.function_registry import function_registry


class RecordingPublisher:
    def __init__(self, args):
        self.args = args

    async def after_save(self, step):
        self.args.events.append(("start", step))
        await self.args.release.wait()
        if self.args.fail:
            raise RuntimeError("publication failed")
        self.args.events.append(("complete", step))


def _args(**kwargs):
    return SimpleNamespace(
        checkpoint_publisher_path="test.publisher",
        async_save=False,
        events=[],
        release=asyncio.Event(),
        fail=False,
        **kwargs
    )


async def test_unconfigured_publisher_has_no_runtime_dependency():
    publisher = CheckpointPublisher(SimpleNamespace())
    await publisher.drain()
    publisher.after_save(0)
    publisher.check()
    await publisher.drain()


async def test_next_save_and_shutdown_wait_for_publication():
    args = _args()
    with function_registry.temporary("test.publisher", RecordingPublisher):
        publisher = CheckpointPublisher(args)
    publisher.after_save(1)
    before_next = asyncio.create_task(publisher.drain())
    await asyncio.sleep(0)
    assert args.events == [("start", 1)]
    assert not before_next.done()
    args.release.set()
    await before_next
    assert args.events == [("start", 1), ("complete", 1)]
    args.release.clear()
    publisher.after_save(2)
    shutdown = asyncio.create_task(publisher.drain())
    await asyncio.sleep(0)
    assert not shutdown.done()
    args.release.set()
    await shutdown
    assert args.events[-1] == ("complete", 2)


async def test_failed_publication_surfaces_at_check_and_drain():
    args = _args()
    args.fail = True
    args.release.set()
    with function_registry.temporary("test.publisher", RecordingPublisher):
        publisher = CheckpointPublisher(args)
    publisher.after_save(1)
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="publication failed"):
        publisher.check()
    with pytest.raises(RuntimeError, match="publication failed"):
        await publisher.drain()


async def test_failed_publication_blocks_the_next_save():
    args = _args()
    args.fail = True
    args.release.set()
    with function_registry.temporary("test.publisher", RecordingPublisher):
        publisher = CheckpointPublisher(args)
    publisher.after_save(1)
    with pytest.raises(RuntimeError, match="publication failed"):
        await publisher.drain()  # what the driver awaits before its next save


def test_async_saves_cannot_publish_before_writers_finish():
    args = _args()
    args.async_save = True
    with function_registry.temporary("test.publisher", RecordingPublisher), pytest.raises(
        ValueError, match="synchronous"
    ):
        CheckpointPublisher(args)
