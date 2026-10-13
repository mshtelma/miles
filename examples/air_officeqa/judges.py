"""Pool of Databricks-served judge models for the OfficeQA path-report reward.

Each call names a routing key; a stable hash of it picks the preferred endpoint, so all
samples of one prompt group are judged by the same model while different prompts spread
over the pool. With ``primary_first`` the first endpoint judges every prompt instead, and
the others only take calls it cannot answer. An endpoint that rate-limits or fails cools down briefly and the call moves
to the next endpoint; one that rejects the request is skipped for that call. If no endpoint
answers before the deadline the caller gets JudgeUnavailable (the reward then treats the
verdict as unknown). Credentials come from the Databricks SDK's default config, which on
AIR is the workspace profile the node already carries.
"""

import hashlib
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from databricks.sdk.core import Config

POOL_CONFIG = Path(__file__).with_name("judges.json")
_RETRYABLE = {408, 429, 500, 502, 503, 504}


class JudgeUnavailable(RuntimeError):
    """No endpoint returned a reply before the call's deadline."""


class _Rejected(Exception):
    """The endpoint refused this request (e.g. too large for the model); retrying will not help."""


@dataclass(frozen=True)
class PoolSettings:
    endpoints: tuple[str, ...]
    max_concurrency: int  # in-flight calls per endpoint
    max_tokens: int
    request_timeout_s: float
    deadline_s: float  # one judge call, across retries and failover
    slot_wait_s: float  # how long to wait for a busy endpoint before trying the next
    cooldown_s: float  # pause after a rate limit or failure without Retry-After
    # Sent to every endpoint when set; newer Claude models reject the parameter with a 400.
    temperature: float | None = None
    # True: every call tries the endpoints in listed order, so the first one judges every
    # prompt and the rest are failover only. False: the routing key spreads prompts over all.
    primary_first: bool = False

    @classmethod
    def from_file(cls, path: Path = POOL_CONFIG) -> "PoolSettings":
        raw = json.loads(path.read_text(encoding="utf-8"))
        settings = cls(**{**raw, "endpoints": tuple(raw["endpoints"])})
        if not settings.endpoints or len(set(settings.endpoints)) != len(settings.endpoints):
            raise ValueError("judge pool needs unique endpoint names")
        return settings


class JudgePool:
    def __init__(
        self,
        settings: PoolSettings,
        *,
        config: Config | None = None,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._settings = settings
        self._config = config or Config()
        self._client = client or httpx.Client(timeout=settings.request_timeout_s)
        self._clock = clock
        self._sleep = sleep
        self._slots = {name: threading.BoundedSemaphore(settings.max_concurrency) for name in settings.endpoints}
        self._cooldown_until = dict.fromkeys(settings.endpoints, 0.0)
        self._lock = threading.Lock()

    def judge(self, key: str, system: str, user: str) -> tuple[str, str]:
        """Return (endpoint, reply text) from the key's endpoint, failing over on errors."""
        deadline = self._clock() + self._settings.deadline_s
        order = self._order(key)
        rejected: set[str] = set()
        while self._clock() < deadline:
            for name in order:
                if name in rejected or self._cooling(name) or self._clock() >= deadline:
                    continue
                try:
                    reply = self._call(name, system, user, deadline)
                except _Rejected:
                    rejected.add(name)
                    continue
                if reply is not None:
                    return name, reply
            if len(rejected) == len(order):
                raise JudgeUnavailable("every judge endpoint rejected the request")
            self._sleep(min(self._settings.cooldown_s, max(0.0, deadline - self._clock())))
        raise JudgeUnavailable(f"no judge endpoint answered within {self._settings.deadline_s:.0f}s")

    def _order(self, key: str) -> list[str]:
        endpoints = list(self._settings.endpoints)
        if self._settings.primary_first:
            return endpoints
        start = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(endpoints)
        return endpoints[start:] + endpoints[:start]

    def _cooling(self, name: str) -> bool:
        with self._lock:
            return self._clock() < self._cooldown_until[name]

    def _cool(self, name: str, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[name] = max(self._cooldown_until[name], self._clock() + seconds)

    def _call(self, name: str, system: str, user: str, deadline: float) -> str | None:
        """The reply text, or None when the endpoint is busy or failed transiently."""
        wait = min(self._settings.slot_wait_s, max(0.0, deadline - self._clock()))
        if not self._slots[name].acquire(timeout=wait):
            return None  # saturated; the next endpoint may be free
        body = {
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self._settings.max_tokens,
        }
        if self._settings.temperature is not None:
            body["temperature"] = self._settings.temperature
        try:
            response = self._client.post(
                f"{self._config.host.rstrip('/')}/serving-endpoints/{name}/invocations",
                headers=self._config.authenticate(),
                json=body,
            )
        except httpx.HTTPError:
            self._cool(name, self._settings.cooldown_s)
            return None
        finally:
            self._slots[name].release()
        if response.status_code in _RETRYABLE:
            self._cool(name, _retry_after(response) or self._settings.cooldown_s)
            return None
        if response.is_error:
            raise _Rejected(f"{name}: HTTP {response.status_code}")
        return _reply_text(response)


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return None


def _reply_text(response: httpx.Response) -> str:
    """The assistant text; an unreadable body yields "" so the verdict parses as unknown."""
    try:
        content = response.json()["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        return ""
    if isinstance(content, list):  # providers that return content blocks
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)
