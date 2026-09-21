"""Fail-fast coordination for rank-local work between distributed collectives."""

from collections.abc import Callable
from dataclasses import dataclass

import torch.distributed as dist

from miles.utils.distributed_utils import get_gloo_group


@dataclass(frozen=True)
class DistributedPhaseFailure:
    """A rank-local failure reported at a distributed phase boundary."""

    rank: int
    message: str


class DistributedPhaseError(RuntimeError):
    """A distributed phase failed on one or more ranks."""

    def __init__(self, phase: str, failures: list[DistributedPhaseFailure]) -> None:
        details = "; ".join(f"rank {failure.rank}: {failure.message}" for failure in failures)
        super().__init__(f"distributed phase {phase!r} failed ({details})")
        self.phase = phase
        self.failures = failures


class DistributedPhaseGate:
    """Share rank-local failures before allowing the next collective."""

    def __init__(self, control_group: dist.ProcessGroup | None = None) -> None:
        self._control_group = control_group

    def run_local_phase(self, phase: str, operation: Callable[[], None]) -> None:
        """Run local work and raise the same error on every rank if it fails."""
        local_error: Exception | None = None
        try:
            operation()
        except Exception as error:
            local_error = error

        if not dist.is_initialized():
            if local_error is not None:
                raise local_error
            return

        group = self._group()
        local_message = None
        if local_error is not None:
            local_message = f"{type(local_error).__name__}: {local_error}"
        messages: list[str | None] = [None] * dist.get_world_size(group=group)
        dist.all_gather_object(messages, local_message, group=group)
        failures = [
            DistributedPhaseFailure(rank=rank, message=message)
            for rank, message in enumerate(messages)
            if message is not None
        ]
        if failures:
            error = DistributedPhaseError(phase, failures)
            if local_error is not None:
                raise error from local_error
            raise error

    def wait_for_all(self) -> None:
        """Wait on the control group after a successful phase."""
        if dist.is_initialized():
            dist.barrier(group=self._group())

    def _group(self) -> dist.ProcessGroup:
        return self._control_group if self._control_group is not None else get_gloo_group()
