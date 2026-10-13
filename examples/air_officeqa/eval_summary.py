"""OfficeQA outcome records and eval-set metrics (pass rates, unbiased pass@k).

The curriculum reads the outcome of each training trajectory from its reward details; the standalone
eval reduces an eval set's rewards with ``summarize_eval_set``. Torch-free, so the eval imports it
without the training stack.
"""

import collections
import json
from math import comb
from typing import Any

# pass@k reported from the per-question samples. With n>=k samples per question, these are the
# unbiased, lower-variance estimates of pass@1 and pass@3 (Chen et al. 2021) -- no dedicated 1- or
# 3-sample run is needed. pass@1 equals the per-sample success rate; questions_solved is pass@n.
PASS_AT_K = (1, 3)


def pass_at_k(rewarded: dict[str, list[float]], k: int) -> float:
    """Unbiased pass@k, macro-averaged over questions with at least ``k`` scored samples.

    For a question with ``n`` scored samples of which ``c`` passed, the chance that none of a random
    ``k``-subset passes is ``C(n-c, k) / C(n, k)``; pass@k is one minus that, averaged over questions.
    Questions with fewer than ``k`` scored (surviving) samples cannot be estimated and are skipped.
    """
    estimates = []
    for rewards in rewarded.values():
        n = len(rewards)
        if n < k:
            continue
        c = sum(1 for reward in rewards if reward > 0)
        estimates.append(1.0 - comb(n - c, k) / comb(n, k))
    return sum(estimates) / len(estimates) if estimates else 0.0


def summarize_eval_set(
    name: str,
    rewarded: dict[str, list[float]],
    categories: collections.Counter,
    lost: int,
) -> dict[str, float]:
    """The ``officeqa_eval/<set>/*`` metrics for one eval set.

    ``rewarded`` maps a question uid to the rewards of its finished trajectories; lost (aborted or
    unscored) trajectories are excluded from the rates and reported as ``lost_trajectories``.
    """
    rewards = [reward for per_question in rewarded.values() for reward in per_question]
    prefix = f"officeqa_eval/{name}/"
    denom = max(1, len(rewards))
    metrics: dict[str, float] = {
        f"{prefix}pass_rate": sum(reward > 0 for reward in rewards) / denom,
        f"{prefix}questions_solved": sum(any(r > 0 for r in rs) for rs in rewarded.values()),
        f"{prefix}lost_trajectories": lost,
    }
    # pass@k is left out when no question has k scored samples, rather than reported as 0.
    metrics.update(
        {
            f"{prefix}pass_at_{k}": pass_at_k(rewarded, k)
            for k in PASS_AT_K
            if any(len(per_question) >= k for per_question in rewarded.values())
        }
    )
    metrics.update({f"{prefix}{category}": count / denom for category, count in categories.items()})
    return metrics


def questions_solved_line(name: str, rewarded: dict[str, list[float]]) -> str:
    """The per-question `solved count` log line, in the curriculum's format."""
    solved = {uid: sum(r > 0 for r in rs) for uid, rs in sorted(rewarded.items())}
    return f"officeqa eval {name}: per-question={json.dumps(solved)}"


def outcome_from_metadata(metadata: dict[str, Any], reward: float) -> dict[str, Any]:
    """Extract outcome dict from agent response metadata and final reward.

    This is the torch-free extraction of what _outcome(sample) computes in the curriculum,
    suitable for standalone eval harnesses that don't have Miles Sample objects.
    """
    metadata = metadata or {}
    details = metadata.get("reward_details") or {}
    reported = metadata.get("stop_reason") == "final_answer"
    return {
        "reward": float(reward or 0.0),
        "category": details.get("category") or ("unscored" if reported else "no_report"),
        "steps": int(details.get("steps", 0)),
        "compute_steps": int(details.get("compute_steps", 0)),
        "compute_steps_with_inputs": int(details.get("compute_steps_with_inputs", 0)),
        "late_inputs": int(details.get("late_inputs", 0)),
        "turns": int(metadata.get("num_turns", 0)),
    }
