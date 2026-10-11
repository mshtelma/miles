"""OfficeQA chat proxy that applies the training session server's context limit and sampling."""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import aiohttp
from aiohttp import web

from examples.air_officeqa.eval.server import FULL_SUPPORT_SAMPLING, SERVED_MODEL_NAME, STOP_TEXTS

logger = logging.getLogger(__name__)

BUDGET_EXHAUSTED = {
    "error": {"code": "context_budget_exhausted", "message": "the trajectory reached its token budget"}
}
_TOKEN_FIELDS = ("max_tokens", "max_completion_tokens")


@dataclass(frozen=True)
class TurnRecord:
    trajectory: str
    status: int
    latency_s: float
    local_prompt_tokens: int | None
    server_prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None


class ChatProxy:
    def __init__(
        self,
        url: str,
        *,
        count_prompt_tokens: Callable[[list, list | None], int],
        max_seq_len: int,
        read_timeout_s: float = 1800.0,
    ) -> None:
        self.url = url
        self.records: list[TurnRecord] = []
        self._count = count_prompt_tokens
        self._max_seq_len = max_seq_len
        self._timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=read_timeout_s)
        self._session: aiohttp.ClientSession | None = None

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024 * 1024)
        app.router.add_post("/t/{trajectory}/v1/chat/completions", self._handle)
        app.on_startup.append(self._open)
        app.on_cleanup.append(self._close)
        return app

    async def _open(self, _app) -> None:
        # A fresh connection per turn, never a kept-alive one: SGLang closes a connection idle for a few seconds,
        # and a turn sent on one the busy loop has not yet seen closed dies with ServerDisconnectedError
        # (an eval lost 7% of its trajectories so). A turn takes seconds, so the extra handshake costs nothing.
        connector = aiohttp.TCPConnector(limit=0, force_close=True)
        self._session = aiohttp.ClientSession(timeout=self._timeout, connector=connector)

    async def _close(self, _app) -> None:
        if self._session is not None:
            await self._session.close()

    async def _handle(self, request: web.Request) -> web.Response:
        trajectory = request.match_info["trajectory"]
        started = time.monotonic()
        body = await request.json()
        body = {**body, "messages": _without_stop_texts(body.get("messages") or [])}
        try:
            local_tokens = await asyncio.to_thread(self._count, body["messages"], body.get("tools"))
        except Exception as error:  # a render failure must be loud, never a silent zero
            logger.exception("prompt token count failed for %s", trajectory)
            return self._record(trajectory, 500, started, None, {"error": {"message": f"render failed: {error}"}})
        remaining = self._max_seq_len - local_tokens
        if remaining <= 0:
            return self._record(trajectory, 422, started, local_tokens, BUDGET_EXHAUSTED)
        status, payload, raw = await self._forward(trajectory, _capped(body, remaining))
        return self._record(trajectory, status, started, local_tokens, payload, raw=raw)

    async def _forward(self, trajectory: str, body: dict) -> tuple[int, dict | None, bytes]:
        """POST to the server; a read timeout is final, as it is for the agent."""
        try:
            async with self._session.post(f"{self.url}/v1/chat/completions", json=body) as response:
                raw = await response.read()
            return response.status, _json_or_none(raw), raw
        except asyncio.TimeoutError:
            return 504, {"error": {"message": "SGLang server timed out"}}, b""
        except aiohttp.ClientError as error:
            logger.warning("SGLang request failed for %s: %r", trajectory, error)
            return 502, {"error": {"message": f"SGLang request failed: {error!r}"}}, b""

    def _record(self, trajectory, status, started, local_tokens, payload, *, raw: bytes | None = None):
        usage = (payload.get("usage") or {}) if isinstance(payload, dict) else {}
        choices = payload.get("choices") if isinstance(payload, dict) else None
        self.records.append(
            TurnRecord(
                trajectory=trajectory,
                status=status,
                latency_s=time.monotonic() - started,
                local_prompt_tokens=local_tokens,
                server_prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=choices[0].get("finish_reason") if choices else None,
            )
        )
        text = raw if raw else json.dumps(payload).encode()
        return web.Response(status=status, body=text, content_type="application/json")

    def turns(self, trajectory: str) -> list[dict]:
        return [asdict(record) for record in self.records if record.trajectory == trajectory]

    def summary(self) -> dict:
        served = [record for record in self.records if record.status == 200]
        latencies = sorted(record.latency_s for record in served)
        return {
            "turns": len(self.records),
            "served_turns": len(served),
            "budget_stops": sum(record.status == 422 for record in self.records),
            "upstream_errors": sum(record.status >= 500 for record in self.records),
            "render_count_mismatch_turns": sum(
                record.server_prompt_tokens is not None and record.server_prompt_tokens != record.local_prompt_tokens
                for record in served
            ),
            "prompt_tokens_max": max((record.local_prompt_tokens or 0 for record in self.records), default=0),
            "latency_p50_s": latencies[len(latencies) // 2] if latencies else 0.0,
        }


def _without_stop_texts(messages: list) -> list:
    """Drop the stop token that ends each assistant turn's text (``no_stop_trim`` keeps it).

    Training appends the next message's tokens right after the generated stop token. Re-rendered from
    text, the template would put that token into the content, ahead of the tool calls, and then render
    ``<|observation|>`` again before the tool results: a doubled stop token training never saw.
    """
    cleaned = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and message.get("role") == "assistant":
            for stop in STOP_TEXTS:
                if content.endswith(stop):
                    message = {**message, "content": content.removesuffix(stop)}
                    break
        cleaned.append(message)
    return cleaned


def _capped(body: dict, remaining: int) -> dict:
    """The session server's rule: cap every token budget the client set, else set one."""
    capped = {**body, "model": body.get("model") or SERVED_MODEL_NAME}
    for field, value in FULL_SUPPORT_SAMPLING.items():
        if capped.get(field) is None:
            capped[field] = value
    present = [field for field in _TOKEN_FIELDS if body.get(field) is not None]
    for field in present:
        capped[field] = min(int(body[field]), remaining)
    if not present:
        capped["max_tokens"] = remaining
    return capped


def _json_or_none(raw: bytes):
    try:
        return json.loads(raw)
    except ValueError:
        return None


async def serve(proxy: ChatProxy, *, host: str = "127.0.0.1", port: int = 0) -> tuple[web.AppRunner, str]:
    """Start the proxy on a local port; returns the runner (for cleanup) and its base URL."""
    runner = web.AppRunner(proxy.app())
    try:
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
    except BaseException:
        await runner.cleanup()
        raise
    bound = runner.addresses[0][1]  # the OS-chosen port when port=0
    return runner, f"http://{host}:{bound}"
