"""Driver checkpoint publication after both trainer and rollout state have been saved.

``--checkpoint-publisher-path`` loads a factory(args) whose object implements async
after_save(rollout_id). One publication may run in the background; the next save and
shutdown drain it, and check() surfaces its errors.
"""

import asyncio

from miles.utils.function_registry import load_function


class CheckpointPublisher:
    def __init__(self, args):
        factory = load_function(getattr(args, "checkpoint_publisher_path", None))
        if factory is not None and args.async_save:
            raise ValueError("checkpoint publishers require synchronous saves")
        self._publisher = factory(args) if factory is not None else None
        self._pending = None

    def after_save(self, rollout_id: int) -> None:
        if self._publisher is not None:
            assert self._pending is None, "previous checkpoint publication was not drained"
            self._pending = asyncio.create_task(self._publisher.after_save(rollout_id))

    def check(self) -> None:
        if self._pending is not None and self._pending.done():
            self._pending.result()

    async def drain(self) -> None:
        if self._pending is not None:
            try:
                await self._pending
            finally:
                self._pending = None
