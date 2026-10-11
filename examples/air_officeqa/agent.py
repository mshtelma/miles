"""Bounded GLM-5.3-Flash agent loop for the Miles TITO session endpoint.

``MILES_TOOL_EXECUTOR`` and ``MILES_REWARD_FN`` name trusted synchronous
``module:function`` adapters. The tool adapter runs work in an isolated sandbox
and enforces its own process deadline; cancelling an asyncio thread alone cannot
kill a tool. The scorer receives the full trajectory and trusted task metadata, and
returns the reward, or an object with the reward under "reward" and details the sample
keeps as ``reward_details``.
"""

import asyncio
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from miles.rollout.generate_utils.agent_errors import EnvironmentFailure, ToolCallError

# Pauses before resending a model request whose connection dropped. A drop usually
# means the session server closed an idle keep-alive socket as the request went out;
# if it did record a turn it could not return, the resent request rolls that turn back.
_MODEL_RETRY_DELAYS_S = (1.0, 5.0)


@dataclass(frozen=True)
class AgentSettings:
    max_turns: int = 64
    max_tools_per_turn: int = 8
    tool_result_max_bytes: int = 200_000
    tool_timeout_s: float = 600.0
    score_timeout_s: float = 120.0
    trajectory_timeout_s: float = 7200.0
    reserve_tokens: int = 16_384
    # A call to one of these tools submits the answer and ends the trajectory unexecuted.
    terminal_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.terminal_tools, (list, tuple)):
            raise ValueError("terminal_tools must be a list of tool names")
        # MILES_AGENT_SETTINGS is JSON, so the names arrive as a list.
        object.__setattr__(self, "terminal_tools", tuple(self.terminal_tools))
        names = self.terminal_tools
        if any(type(name) is not str or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("terminal_tools must be unique nonempty tool names")
        counts = (self.max_turns, self.max_tools_per_turn, self.tool_result_max_bytes)
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("agent counts and byte cap must be positive integers")
        deadlines = (self.tool_timeout_s, self.score_timeout_s, self.trajectory_timeout_s)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 for value in deadlines):
            raise ValueError("agent deadlines must be positive and finite")
        if type(self.reserve_tokens) is not int or self.reserve_tokens < 0:
            raise ValueError("reserve_tokens must be a nonnegative integer")


class _ContextBudgetExhausted(Exception):
    """Typed session admission failure; the policy has reached its token budget."""


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Drive a single trajectory, returning a score tied to a response ID."""
    settings = AgentSettings(**json.loads(os.environ.get("MILES_AGENT_SETTINGS", "{}")))
    run_tool = _load_env_function("MILES_TOOL_EXECUTOR")
    score = _load_env_function("MILES_REWARD_FN")
    try:
        return await asyncio.wait_for(
            _drive(base_url, prompt, request_kwargs or {}, metadata or {}, settings, run_tool, score),
            timeout=settings.trajectory_timeout_s,
        )
    except TimeoutError as error:
        raise EnvironmentFailure("trajectory deadline exceeded") from error


async def _drive(base_url, prompt, request_kwargs, metadata, settings, run_tool, score) -> dict[str, Any]:
    messages = _initial_messages(prompt)
    tools = _validated_tools(metadata.get("tools"))
    allowed_tools = {tool["function"]["name"] for tool in tools}
    if not isinstance(metadata.get("tool_context", {}), dict):
        raise ValueError("tool_context must be an object")
    if any(type(metadata.get(key)) is not str or not metadata[key] for key in ("session_id", "attempt_id")):
        raise ValueError("trusted session_id and attempt_id must be nonempty text")
    budget = metadata.get("max_seq_len")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError("max_seq_len must be a positive integer")
    request = {key: value for key, value in request_kwargs.items() if key not in {"tools", "messages"}}
    request.update(stream=False, n=1)
    stop_reason, tool_time, last_record_id, num_turns = "max_turns", 0.0, None, 0
    model_responses = responses_with_usage = model_retries = 0
    timeout = httpx.Timeout(connect=30.0, read=1800.0, write=60.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for _ in range(settings.max_turns):
            num_turns += 1
            try:
                choice, used_tokens, last_record_id, retries = await _complete(
                    client, base_url, messages=messages, tools=tools, request=request
                )
            except _ContextBudgetExhausted:
                stop_reason = "budget"
                break
            model_retries += retries
            model_responses += 1
            responses_with_usage += used_tokens is not None
            messages.append(choice["message"])
            reason = _stop_reason(
                choice,
                used_tokens=used_tokens,
                budget=budget,
                reserve=settings.reserve_tokens,
                terminal_tools=settings.terminal_tools,
            )
            if reason is not None:
                stop_reason = reason
                break
            calls = choice["message"]["tool_calls"]
            if len(calls) > settings.max_tools_per_turn:
                stop_reason = "tool_limit"
                break
            started = time.monotonic()
            for call in calls:
                content = await _tool_result(
                    run_tool, call=call, metadata=metadata, allowed_tools=allowed_tools, settings=settings
                )
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
            tool_time += time.monotonic() - started
    reward, details = 0.0, None
    if stop_reason == "final_answer":
        reward, details = await _score(score, messages=messages, metadata=metadata, timeout_s=settings.score_timeout_s)
    result = {
        "reward": reward,
        "reward_record_id": last_record_id,
        "num_turns": num_turns,
        "stop_reason": stop_reason,
        "agent_metrics": {
            "total_tool_time": tool_time,
            "model_responses": model_responses,
            "responses_with_usage": responses_with_usage,
            "model_retries": model_retries,
        },
    }
    if details is not None:
        result["reward_details"] = details
    return result


def _initial_messages(prompt: Any) -> list[dict[str, Any]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt or any(not isinstance(message, dict) for message in prompt):
        raise ValueError("agent prompt must be raw text or a nonempty message list")
    return copy.deepcopy(prompt)


def _validated_tools(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("tools must be a list in task metadata")
    tools = copy.deepcopy(value)
    names = []
    for tool in tools:
        if not isinstance(tool, dict) or not isinstance(tool.get("function"), dict):
            raise ValueError("tool schema must have a function object")
        name = tool["function"].get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool function name must be nonempty text")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("tool names must be unique")
    return tools


async def _post_completion(client, url: str, payload: dict) -> tuple[httpx.Response, int]:
    """POST once, resending after a dropped connection; a read timeout is final."""
    for retries, delay in enumerate((*_MODEL_RETRY_DELAYS_S, None)):
        try:
            return await client.post(url, json=payload), retries
        except httpx.ReadTimeout as error:
            raise EnvironmentFailure("model transport failure") from error
        except httpx.TransportError as error:
            if delay is None:
                raise EnvironmentFailure("model transport failure") from error
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _complete(client, base_url, *, messages, tools, request):
    response, retries = await _post_completion(
        client, f"{base_url}/v1/chat/completions", {"messages": messages, "tools": tools, **request}
    )
    if response.status_code == 422:
        try:
            error_body = response.json().get("error", {})
        except (ValueError, AttributeError):
            error_body = {}
        if isinstance(error_body, dict) and error_body.get("code") == "context_budget_exhausted":
            raise _ContextBudgetExhausted
    if response.is_error:
        raise EnvironmentFailure(f"model status {response.status_code}: {response.text[:512]}")
    body = response.json()
    if not isinstance(body, dict) or len(body.get("choices", [])) != 1 or not isinstance(body.get("id"), str):
        raise ValueError("unexpected model response schema")
    choice = body["choices"][0]
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        raise ValueError("model response has no assistant message")
    message = choice["message"]
    if message.get("role") != "assistant":
        raise ValueError("model response role is not assistant")
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or any(not _valid_call(call) for call in calls):
        raise ValueError("model response contains an invalid tool call")
    message["tool_calls"] = calls
    used_tokens = (body.get("usage") or {}).get("total_tokens")
    if used_tokens is not None and (type(used_tokens) is not int or used_tokens < 0):
        raise ValueError("model usage.total_tokens must be a nonnegative integer")
    return choice, used_tokens, body["id"], retries


def _valid_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and isinstance(call.get("id"), str)
        and bool(call["id"])
        and isinstance(call.get("function"), dict)
        and isinstance(call["function"].get("name"), str)
    )


def _stop_reason(
    choice: dict[str, Any],
    *,
    used_tokens: int | None,
    budget: int | None,
    reserve: int,
    terminal_tools: tuple[str, ...] = (),
) -> str | None:
    if budget is not None and used_tokens is not None and used_tokens > budget:
        raise EnvironmentFailure("server exceeded the hard token budget")
    finish = choice.get("finish_reason")
    if finish == "abort":
        raise EnvironmentFailure("engine aborted the request")
    if finish == "length":
        return "length"
    if finish not in {"stop", "tool_calls"}:
        raise ValueError(f"unsupported finish reason: {finish!r}")
    calls = choice["message"]["tool_calls"]
    if (finish == "tool_calls") != bool(calls):
        raise ValueError("finish reason and assistant tool calls disagree")
    if not calls:
        return "final_answer"
    # Checked before the reserve: a submission needs no further turn, so it counts even near the budget.
    if any(call["function"]["name"] in terminal_tools for call in calls):
        return "final_answer"
    if budget is not None and used_tokens is not None and used_tokens + reserve > budget:
        return "budget"
    return None


def _bounded_tool_text(value: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError("tool adapters must return text")
    # errors="replace": adapters may return lone surrogates from surrogateescape-decoded output.
    raw = value.encode("utf-8", errors="replace")
    marker = b"\n[tool result truncated]"
    if len(raw) <= limit:
        return raw.decode("utf-8")
    if limit < len(marker):
        raise ValueError("tool byte cap is smaller than the truncation marker")
    return raw[: limit - len(marker)].decode("utf-8", errors="ignore") + marker.decode("ascii")


def _reject_json_constant(value: str):
    raise ToolCallError(f"non-JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ToolCallError("non-finite tool argument")
    return number


def _parse_tool_arguments(raw_arguments: str) -> dict:
    """Policy-generated text: every parse failure is a policy-visible tool error, never a crash."""
    try:
        arguments = json.loads(raw_arguments, parse_constant=_reject_json_constant, parse_float=_finite_float)
    except ToolCallError:
        raise
    except (ValueError, RecursionError) as error:  # JSONDecodeError, the int-digit limit, deep nesting
        raise ToolCallError(f"invalid tool arguments: {type(error).__name__}") from error
    if not isinstance(arguments, dict):
        raise ToolCallError("tool arguments must be a JSON object")
    return arguments


def _session_ref(session_id: str, attempt_id: str) -> str:
    """Give sandbox adapters a stable trace key without the session bearer ID."""
    return hashlib.sha256(f"{session_id}\x00{attempt_id}".encode()).hexdigest()


async def _tool_result(run_tool, *, call, metadata, allowed_tools, settings) -> str:
    try:
        name = call["function"]["name"]
        if name not in allowed_tools:
            raise ToolCallError("unknown tool")
        raw_arguments = call["function"].get("arguments") or "{}"
        if not isinstance(raw_arguments, str):
            raise ToolCallError("tool arguments must be a JSON string")
        arguments = _parse_tool_arguments(raw_arguments)
        # The adapter enforces this deadline at the sandbox boundary as well.
        context = {
            "task": copy.deepcopy(metadata.get("tool_context", {})),
            "session_ref": _session_ref(metadata["session_id"], metadata["attempt_id"]),
            "attempt_id": metadata["attempt_id"],
            "tool_call_id": call["id"],
            "tool_deadline_s": settings.tool_timeout_s,
        }
        # Keep blocking sandbox calls off asyncio's default pool, which also decodes
        # samples. Queueing behind unrelated work must not consume the tool deadline.
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm53-tool")
        try:
            future = asyncio.get_running_loop().run_in_executor(executor, run_tool, name, arguments, context)
            result = await asyncio.wait_for(future, settings.tool_timeout_s)
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return _bounded_tool_text(result, settings.tool_result_max_bytes)
    except ToolCallError as error:
        return _bounded_tool_text(json.dumps({"error": str(error)}), settings.tool_result_max_bytes)
    except TimeoutError as error:
        raise EnvironmentFailure("sandbox adapter exceeded its deadline") from error


async def _score(score, *, messages, metadata, timeout_s) -> tuple[float, dict | None]:
    """(reward, details): a scorer returns the reward, or an object holding it under "reward"."""
    # A judge-backed scorer blocks for tens of seconds; like tools, keep it off asyncio's
    # default pool so many concurrent scores cannot starve sample decoding.
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="glm53-score")
    try:
        future = asyncio.get_running_loop().run_in_executor(executor, score, messages, metadata)
        result = await asyncio.wait_for(future, timeout_s)
    except TimeoutError as error:
        raise EnvironmentFailure("scorer deadline exceeded") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    details = None
    if isinstance(result, dict):
        details = {key: item for key, item in result.items() if key != "reward"}
        result = result.get("reward")
    if isinstance(result, bool) or not isinstance(result, (int, float)) or not math.isfinite(result):
        raise ValueError("scorer returned a non-finite reward")
    return float(result), details


def _load_env_function(env_var: str):
    module_name, separator, function_name = os.environ[env_var].partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"{env_var} must name a trusted module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function) or inspect.iscoroutinefunction(function):
        raise TypeError(f"{env_var} must be a synchronous bounded adapter")
    return function
