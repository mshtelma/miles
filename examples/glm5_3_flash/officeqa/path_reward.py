"""OfficeQA reward: a correct answer whose cited path the judge confirms.

``score`` is the ``MILES_REWARD_FN`` (settings.py). It rebuilds the episode's observations from the
trajectory and runs the checks in path_report.py. The reward is 1 when the report is valid, every cited
observation was delivered, no looked-up figure is fabricated, the answer matches gold and the judge
finds the path supported; anything else is 0, including an unresolved verdict. ``score`` returns the
reward with the details behind it, and logs every outcome.

  * The reward is 1, not the judge's graded path score, so every all-correct group has zero advantage
    instead of training on the judge's grading.
  * The judge is a pool of Databricks-served models (judges.py), routed per prompt so a group shares
    one judge, and skipped when the answer is wrong, since the reward is 0 either way.
  * Answers are checked by answer_check.py, which honours scale words, signs and list length, and
    needs the answer to state only its final value.
  * A compute step whose declared inputs arrived after it was generated is counted, not rejected:
    ``depends_on`` is optional, so rejecting it taught the policy to stop declaring inputs.
  * A looked-up figure must appear in a delivered observation, also in a step with ``depends_on``.
"""

import dataclasses
import json
import logging
import re
import threading
import time

from examples.glm5_3_flash.officeqa import answer_check, judges, path_report
from examples.glm5_3_flash.officeqa.adapters import ANSWER_TOLERANCE, SUBMIT_TOOL

logger = logging.getLogger(__name__)

MAX_HISTORY_BYTES = 120_000  # tool history shown to the judge, about 30K tokens
VERDICT_RETRIES = 2  # re-ask on an unparseable verdict
_LATE_INPUT = "impossible_chronology"

_OBSERVATION = re.compile(r"\[observation (obs_(\d+))\]\n")
_pool_lock = threading.Lock()
_pool: judges.JudgePool | None = None


def score(messages: list[dict], metadata: dict) -> dict:
    """The reward and the outcome behind it: status, category, reason and report shape."""
    started = time.monotonic()
    record = episode_record(messages, metadata)
    report = path_report.parse_report(record["terminal_text"])
    ledger = path_report.EpisodeLedger.from_record(record)
    refs = path_report.check_references(report, ledger)
    late_inputs = sum(issue.code == _LATE_INPUT for issue in refs.issues)
    kept = [issue for issue in refs.issues if issue.code != _LATE_INPUT]
    refs = path_report.RefCheck(not kept, kept)
    values = check_cited_values(report, ledger)
    correct, answer_reason = _answer_check(report, str(metadata["reference_answer"]))
    verdict = None
    if report.ok and refs.valid and values.valid and correct is False:
        status, path_score, reason = path_report.ZERO, 0.0, f"{answer_reason} (judge skipped)"
        category = {answer_check.NOT_LEADING: "not_leading", answer_check.EXTRA: "extra_values"}.get(
            answer_reason, "wrong"
        )
    else:
        if report.ok and refs.valid and values.valid and correct:
            verdict = _judge(support_request(record, report, ledger), key=str(metadata["uid"]))
        outcome = path_report.assess(report=report, ref=refs, verdict=verdict, answer_correct=correct, values=values)
        status, reason = outcome.status, outcome.reason
        path_score = outcome.score if outcome.status == path_report.SCORED else 0.0
        category = _category(report, refs, values, verdict, status)
    reward = 1.0 if status == path_report.SCORED else 0.0
    compute_steps = [
        step for step in report.steps if (obs := ledger.get(step.observation)) is not None and obs.is_compute
    ]
    details = {
        "status": status,
        "category": category,
        "reason": reason[:160],
        "judge": (verdict or {}).get("endpoint"),
        "path_score": path_score,
        "steps": len(report.steps),
        "compute_steps": len(compute_steps),
        "compute_steps_with_inputs": sum(bool(step.depends_on) for step in compute_steps),
        "late_inputs": late_inputs,
    }
    logger.info(
        "officeqa path-report uid=%s status=%s reward=%.3f judge=%s reason=%s steps=%d compute=%d "
        "declared_inputs=%d late_inputs=%d elapsed=%.1fs",
        metadata["uid"],
        status,
        reward,
        details["judge"],
        details["reason"],
        details["steps"],
        details["compute_steps"],
        details["compute_steps_with_inputs"],
        late_inputs,
        time.monotonic() - started,
    )
    return {"reward": reward, **details}


def check_cited_values(report: path_report.ParsedReport, ledger: path_report.EpisodeLedger) -> path_report.ValueCheck:
    """The fabricated-value check over every lookup step, those with ``depends_on`` too."""
    if not report.ok:
        return path_report.ValueCheck(True, [])
    steps = [dataclasses.replace(step, depends_on=[]) for step in report.steps]
    return path_report.check_claimed_values(dataclasses.replace(report, steps=steps), ledger)


def _category(report, refs, values, verdict, status) -> str:
    """The first failure ``path_report.assess`` applied, as a short label for metrics."""
    if status == path_report.SCORED:
        return "correct"
    if status in (path_report.ABSTENTION, path_report.UNKNOWN):
        return status
    if not report.ok:
        return "malformed"
    if not refs.valid:
        return "invalid_path"
    if not values.valid:
        return "fabricated"
    if (verdict or {}).get("path_status") == path_report.UNSUPPORTED:
        return "unsupported"
    return "wrong"


def episode_record(messages: list[dict], metadata: dict) -> dict:
    """The episode record the checks read, rebuilt from the trajectory.

    Observation ids come from the markers ``adapters.run_tool`` put on tool messages, never
    from model text. Every tool result precedes the terminal assistant turn, so each one
    was delivered to the request after the one that issued it.
    """
    calls, observations, request = {}, [], -1
    for message in messages:
        if message.get("role") == "assistant":
            request += 1
            for call in message.get("tool_calls") or []:
                calls[call["id"]] = (call["function"]["name"], call["function"].get("arguments"), request)
        elif message.get("role") == "tool" and message.get("tool_call_id") in calls:
            content = message.get("content") or ""
            marker = _OBSERVATION.match(content)
            if marker is None:
                continue  # a policy-visible tool error, not an observation
            name, raw_arguments, issued_by = calls[message["tool_call_id"]]
            text = content[marker.end() :]
            observations.append(
                {
                    "observation_id": marker.group(1),
                    "order": int(marker.group(2)),
                    "tool": name,
                    "args": _arguments(raw_arguments),
                    "outcome": _outcome(text),
                    "delivered_text": text,
                    "generated_by_request": issued_by,
                    "delivered_to_request": issued_by + 1,
                }
            )
    return {
        "episode_id": str(metadata["uid"]),
        "question": metadata.get("question") or _first_user_text(messages),
        "question_requirements": metadata.get("question_requirements") or "",
        "observations": observations,
        "terminal_text": _terminal_text(messages),
    }


def support_request(record: dict, report: path_report.ParsedReport, ledger: path_report.EpisodeLedger) -> dict:
    return path_report.build_support_request(
        record["question"], record["question_requirements"], report, ledger, max_history_bytes=MAX_HISTORY_BYTES
    )


def _judge(request: dict, *, key: str) -> dict:
    system, user = path_report.render_support_prompt(request)
    try:
        pool = _judge_pool()
    except Exception as error:  # noqa: BLE001 - a missing judge fails closed, never aborts the group
        logger.exception("officeqa judge pool unavailable")
        return {
            "status": "unknown",
            "reason": f"judge pool unavailable: {type(error).__name__}",
            "kind": "judge_fault",
        }
    verdict: dict = {}
    for _ in range(1 + VERDICT_RETRIES):
        try:
            endpoint, reply = pool.judge(key, system, user)
        except judges.JudgeUnavailable as error:
            return {"status": "unknown", "reason": str(error), "kind": "judge_fault"}
        verdict = {**path_report.parse_support_verdict(reply), "endpoint": endpoint}
        if verdict["status"] == "ok":
            break
    return verdict


def _judge_pool() -> judges.JudgePool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = judges.JudgePool(judges.PoolSettings.from_file())
        return _pool


def _answer_check(report: path_report.ParsedReport, gold: str) -> tuple[bool | None, str]:
    """(correct, reason); correct is None when there is no answer to score."""
    if not report.ok or not report.answer:
        return None, "no answer"
    try:
        return answer_check.check(report.answer, gold, tolerance=ANSWER_TOLERANCE)
    except (ValueError, TypeError, ArithmeticError):  # a scorer fault must not fabricate a label
        return None, "answer could not be scored"


def _terminal_text(messages: list[dict]) -> str:
    """The submit_report arguments as report JSON; "" when the trajectory did not submit.

    Tool-call parsers often serialize the nested ``path`` list as a JSON string; decode it, and pass
    anything else through for the parser to judge.
    """
    last = messages[-1] if messages else {}
    if last.get("role") != "assistant":
        return ""
    for call in last.get("tool_calls") or []:
        if call["function"]["name"] != SUBMIT_TOOL:
            continue
        raw = call["function"].get("arguments") or "{}"
        try:
            report = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(report, dict) and isinstance(report.get("path"), str):
            try:
                report = {**report, "path": json.loads(report["path"])}
            except ValueError:
                pass
        return json.dumps(report, ensure_ascii=False)
    return ""


def _arguments(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _outcome(text: str) -> str:
    """A coarse delivery tag: error, empty or ok (never semantic support)."""
    if not text.strip():
        return "empty"
    stripped = text.lstrip()
    if stripped.startswith("Error:"):
        return "error"
    if stripped.startswith("No ") and re.search(r"found|match|document", stripped[:80]):
        return "empty"
    return "ok"


def _first_user_text(messages: list[dict]) -> str:
    return next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
