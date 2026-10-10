"""Run the training OfficeQA agent over eval sets and reduce per-question rewards."""

import asyncio
import collections
import copy
import json
import logging
import math
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from examples.glm5_3_flash.officeqa.eval_summary import (
    outcome_from_metadata,
    questions_solved_line,
    summarize_eval_set,
)
from miles.rollout.generate_utils.agent_errors import EnvironmentFailure

logger = logging.getLogger(__name__)

# The training rollout's sampling, as build_chat_request_kwargs maps it: temperature,
# top_p and top_k from rollout_*, the 16K per-turn cap as max_tokens, no stop strings, special tokens kept.
REQUEST_KWARGS = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_tokens": 16384,
    "skip_special_tokens": False,
    "no_stop_trim": True,
    "spaces_between_special_tokens": False,
}

AgentRun = Callable[..., Awaitable[dict]]


@dataclass
class SetOutcome:
    rewarded: dict[str, list[float]] = field(default_factory=lambda: collections.defaultdict(list))
    categories: collections.Counter = field(default_factory=collections.Counter)
    lost: int = 0
    total: int = 0
    trajectories: list[dict] = field(default_factory=list)


def load_rows(path: Path) -> list[dict]:
    """Eval rows as prepare_data writes them: prompt, label and metadata (with uid, tools, tool_context)."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


async def run_sets(
    sets: dict[str, list[dict]],
    *,
    agent_run: AgentRun,
    proxy,
    proxy_url: str,
    samples_per_prompt: int,
    max_concurrent: int,
    max_seq_len: int,
    attempt_id: str,
) -> dict[str, SetOutcome]:
    """Every (set, question, sample) trajectory under one shared concurrency limit."""
    limit = asyncio.Semaphore(max_concurrent)
    outcomes = {name: SetOutcome() for name in sets}
    jobs = [
        _run_one(
            name,
            row,
            sample,
            outcome=outcomes[name],
            limit=limit,
            agent_run=agent_run,
            proxy=proxy,
            proxy_url=proxy_url,
            max_seq_len=max_seq_len,
            attempt_id=attempt_id,
        )
        for name, rows in sets.items()
        for row in rows
        for sample in range(samples_per_prompt)
    ]
    await asyncio.gather(*jobs)
    return outcomes


async def _run_one(name, row, sample, *, outcome, limit, agent_run, proxy, proxy_url, max_seq_len, attempt_id):
    uid = row["metadata"]["uid"]
    trajectory = f"{name}-{uid}-{sample}-{uuid.uuid4().hex[:8]}"
    metadata = {
        **copy.deepcopy(row["metadata"]),
        "max_seq_len": max_seq_len,
        "session_id": uuid.uuid4().hex,
        "attempt_id": attempt_id,
    }
    record = {"set": name, "uid": uid, "sample": sample, "trajectory": trajectory}
    outcome.total += 1
    async with limit:
        started = time.monotonic()
        try:
            result = await agent_run(
                base_url=f"{proxy_url}/t/{trajectory}",
                prompt=copy.deepcopy(row["prompt"]),
                request_kwargs=dict(REQUEST_KWARGS),
                metadata=metadata,
            )
            record.update(_scored(result, outcome, uid))
        except EnvironmentFailure as error:
            record.update(status="lost", error=f"environment: {error}")
        except Exception as error:  # an agent bug is lost too, but counted under its own class
            logger.exception("agent failed on %s", trajectory)
            record.update(status="lost", error=f"{type(error).__name__}: {error}")
        record["wall_s"] = time.monotonic() - started
    if record["status"] == "lost":
        outcome.lost += 1
    record["turns"] = proxy.turns(trajectory)
    outcome.trajectories.append(record)


def _scored(result: dict, outcome: SetOutcome, uid: str) -> dict:
    reward = float(result["reward"])
    if not math.isfinite(reward):
        raise ValueError("the agent returned a non-finite reward")
    # The agent result carries what the curriculum reads off a sample's metadata.
    summary = outcome_from_metadata(result, reward)
    outcome.rewarded[uid].append(reward)
    outcome.categories[summary["category"]] += 1
    return {
        "status": "scored",
        "reward": reward,
        "category": summary["category"],
        "stop_reason": result.get("stop_reason"),
        "num_turns": result.get("num_turns"),
        "agent_metrics": result.get("agent_metrics"),
    }


def set_metrics(name: str, outcome: SetOutcome) -> dict[str, float]:
    """Option A's ``officeqa_eval/<set>/*`` metrics, including the lost-as-zero floor."""
    metrics = summarize_eval_set(name, dict(outcome.rewarded), outcome.categories, outcome.lost)
    passed = sum(reward > 0 for rewards in outcome.rewarded.values() for reward in rewards)
    metrics[f"officeqa_eval/{name}/pass_rate_lost_as_zero"] = passed / max(1, outcome.total)
    logger.info(questions_solved_line(name, dict(outcome.rewarded)))
    return metrics


def lost_fraction(outcomes: dict[str, SetOutcome]) -> float:
    total = sum(outcome.total for outcome in outcomes.values())
    return sum(outcome.lost for outcome in outcomes.values()) / max(1, total)


def write_outputs(out_dir: Path, *, model: dict, summary: dict, outcomes: dict[str, SetOutcome]) -> None:
    """eval_results.json (every reward per question), summary.json and trajectories.jsonl."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {"model": model, "results": {name: dict(outcome.rewarded) for name, outcome in outcomes.items()}}
    (out_dir / "eval_results.json").write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    rows = (record for outcome in outcomes.values() for record in outcome.trajectories)
    (out_dir / "trajectories.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
