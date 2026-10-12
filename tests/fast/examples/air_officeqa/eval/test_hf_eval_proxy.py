"""The proxy reproduces the training session server's token budget, sampling and stop-token handling."""

import asyncio
import json
import re
import socket

import aiohttp
from aiohttp import web

from examples.air_officeqa.eval import proxy

# GLM-5.3-Flash's stop tokens (generation_config.json eos_token_id 154829, 154827, 154820).
_GLM_STOPS = ("<|observation|>", "<|user|>", "<|endoftext|>")
_OK = {
    "id": "chatcmpl-1",
    "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}


class _Upstream:
    """A fake SGLang server that records what it received."""

    def __init__(self, *, status: int = 200, payload: dict | None = None) -> None:
        self.received: list[dict] = []
        self.status = status
        self.payload = payload if payload is not None else _OK

    async def start(self) -> str:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        return f"http://127.0.0.1:{self.runner.addresses[0][1]}"

    async def _handle(self, request: web.Request) -> web.Response:
        self.received.append(await request.json())
        return web.json_response(self.payload, status=self.status)


class _KeepAliveExpiringUpstream:
    """A server that drops a kept-alive connection when a second request arrives on it.

    SGLang (uvicorn) closes a connection idle for a few seconds, and a request sent on it right then dies with
    ServerDisconnectedError before the server sees it.
    """

    def __init__(self) -> None:
        self.served = 0
        self.dropped = 0

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._connection, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        await reader.readexactly(int(re.search(rb"content-length: *(\d+)", head, re.IGNORECASE).group(1)))
        body = json.dumps(_OK).encode()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body))
        writer.write(body)
        await writer.drain()
        self.served += 1
        if await reader.read(1):  # the next turn came in on the expired connection
            self.dropped += 1
        writer.close()


def _dead_url() -> str:
    with socket.socket() as probe:  # a port nothing listens on
        probe.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{probe.getsockname()[1]}"


async def _post(url: str, trajectory: str, body: dict) -> tuple[int, dict]:
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/t/{trajectory}/v1/chat/completions", json=body) as response:
            return response.status, await response.json()


async def _serve(server_url: str, *, tokens: int = 10, max_seq_len: int = 150_000):
    chat = proxy.ChatProxy(
        server_url, count_prompt_tokens=lambda messages, tools: tokens, max_seq_len=max_seq_len, stop_texts=_GLM_STOPS
    )
    runner, url = await proxy.serve(chat)
    return chat, runner, url


async def test_turn_budget_is_capped_to_the_context_left_and_the_body_passes_through() -> None:
    upstream = _Upstream()
    chat, runner, url = await _serve(await upstream.start(), tokens=140_000)
    try:
        status, body = await _post(url, "t1", {"messages": [{"role": "user", "content": "q"}], "max_tokens": 16384})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    assert status == 200 and body == _OK
    sent = upstream.received[0]
    assert sent["max_tokens"] == 10_000 and sent["model"] == proxy.SERVED_MODEL_NAME
    assert sent["messages"] == [{"role": "user", "content": "q"}]
    assert chat.turns("t1")[0]["local_prompt_tokens"] == 140_000


async def test_no_budget_left_answers_the_session_servers_422_without_calling_a_server() -> None:
    upstream = _Upstream()
    chat, runner, url = await _serve(await upstream.start(), tokens=150_000)
    try:
        status, body = await _post(url, "t1", {"messages": [], "max_tokens": 16384})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    assert status == 422 and body["error"]["code"] == "context_budget_exhausted"
    assert upstream.received == [] and chat.summary()["budget_stops"] == 1


async def test_a_request_without_a_budget_gets_the_context_left() -> None:
    upstream = _Upstream()
    _, runner, url = await _serve(await upstream.start(), tokens=149_000)
    try:
        await _post(url, "t1", {"messages": []})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    assert upstream.received[0]["max_tokens"] == 1_000


async def test_prior_assistant_turns_lose_their_trailing_stop_token_before_rendering() -> None:
    # Training continues right after the generated <|observation|>; re-rendered, it would appear twice.
    upstream = _Upstream()
    seen = []
    chat = proxy.ChatProxy(
        await upstream.start(),
        count_prompt_tokens=lambda messages, tools: seen.append(messages) or 10,
        max_seq_len=150_000,
        stop_texts=_GLM_STOPS,
    )
    runner, url = await proxy.serve(chat)
    call = {"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "<|observation|>", "reasoning_content": "r", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "keeps <|observation|> as data"},
        {"role": "assistant", "content": "done<|user|>"},
    ]
    try:
        await _post(url, "t1", {"messages": history})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    sent = upstream.received[0]["messages"]
    assert seen[0] == sent  # the count renders exactly what the server receives
    assert [m["content"] for m in sent] == ["q", "", "keeps <|observation|> as data", "done"]
    assert sent[1]["tool_calls"] == [call] and sent[1]["reasoning_content"] == "r"


async def test_missing_top_p_and_top_k_get_full_support_and_set_ones_pass_through() -> None:
    # The model's generation-config top_p would leave the training sampling and crash the server.
    upstream = _Upstream()
    _, runner, url = await _serve(await upstream.start())
    try:
        await _post(url, "t1", {"messages": [], "temperature": 1.0})
        await _post(url, "t2", {"messages": [], "top_p": 0.9, "top_k": 20})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    filled, kept = upstream.received
    assert (filled["top_p"], filled["top_k"], filled["temperature"]) == (1.0, -1, 1.0)
    assert (kept["top_p"], kept["top_k"]) == (0.9, 20)


async def test_every_turn_gets_its_own_connection_so_an_expired_keep_alive_never_drops_it() -> None:
    upstream = _KeepAliveExpiringUpstream()
    _, runner, url = await _serve(await upstream.start())
    try:
        statuses = [(await _post(url, "a", {"messages": []}))[0] for _ in range(3)]
    finally:
        await runner.cleanup()
        upstream.server.close()
        await upstream.server.wait_closed()

    assert statuses == [200, 200, 200]
    assert (upstream.served, upstream.dropped) == (3, 0)


async def test_an_unreachable_server_is_a_502_the_agent_reports_as_lost() -> None:
    chat, runner, url = await _serve(_dead_url())
    try:
        status, body = await _post(url, "a", {"messages": []})
    finally:
        await runner.cleanup()

    assert status == 502 and "SGLang request failed" in body["error"]["message"]
    assert chat.summary()["upstream_errors"] == 1


async def test_a_render_failure_is_loud_not_a_zero_token_count() -> None:
    upstream = _Upstream()

    def broken(messages, tools):
        raise ValueError("template error")

    chat = proxy.ChatProxy(
        await upstream.start(), count_prompt_tokens=broken, max_seq_len=150_000, stop_texts=_GLM_STOPS
    )
    runner, url = await proxy.serve(chat)
    try:
        status, body = await _post(url, "a", {"messages": []})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    assert status == 500 and "template error" in body["error"]["message"] and upstream.received == []


async def test_server_errors_pass_through_and_count_mismatches_are_reported() -> None:
    upstream = _Upstream(payload={**_OK, "usage": {"prompt_tokens": 11, "completion_tokens": 2}})
    chat, runner, url = await _serve(await upstream.start(), tokens=10)
    try:
        await _post(url, "a", {"messages": []})
        upstream.status, upstream.payload = 400, {"error": {"message": "bad"}}
        status, body = await _post(url, "b", {"messages": []})
    finally:
        await runner.cleanup()
        await upstream.runner.cleanup()

    assert status == 400 and body == {"error": {"message": "bad"}}
    summary = chat.summary()
    assert summary["render_count_mismatch_turns"] == 1 and summary["served_turns"] == 1
