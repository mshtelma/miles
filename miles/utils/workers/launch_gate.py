import functools

import httpx

from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.retry_utils import retry_until_deadline
from miles.utils.workers.argv_utils import _record_field_names

GATE_PORT_NAME = "gate"

_INITIAL_DELAY_SECONDS = 1.0
_MAX_DELAY_SECONDS = 5.0
_ATTEMPT_TIMEOUT_SECONDS = 30.0


async def activate_launch_gate(gate_url: str, timeout: float = 1800.0) -> None:
    async def _activate(remaining_seconds: float) -> None:
        response = await GeneralHttpClientProvider.client().post(
            f"{gate_url}/gate/activate",
            json={},
            timeout=min(_ATTEMPT_TIMEOUT_SECONDS, remaining_seconds),
        )
        response.raise_for_status()

    await retry_until_deadline(
        _activate,
        total_seconds=timeout,
        retry_on=(httpx.HTTPError, OSError),
        initial_delay=_INITIAL_DELAY_SECONDS,
        max_delay=_MAX_DELAY_SECONDS,
        log_fields=dict(gate_url=gate_url),
    )


@functools.cache
def sglang_serves_launch_gate() -> bool:
    """Whether the installed SGLang holds engine startup behind the gate port Miles allocates.

    Miles drops server arguments that an SGLang build does not define, so a build without
    ``gated_launch_port`` starts at once and never serves the gate.
    """
    # Deferred: this module is on the light worker import path, and ServerArgs imports torch.
    from sglang.srt.server_args import ServerArgs

    return "gated_launch_port" in _record_field_names(ServerArgs)
