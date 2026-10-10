"""The driver runs every trajectory as the rollout would and reduces them like an in-run sweep."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from examples.glm5_3_flash.officeqa.eval import driver

from miles.rollout.generate_utils.agent_errors import EnvironmentFailure


def _row(uid: str) -> dict:
    return {
        "prompt": [{"role": "system", "content": "s"}, {"role": "user", "content": f"question {uid}"}],
        "label": "42",
        "metadata": {"uid": uid, "reference_answer": "42", "tools": [], "tool_context": {"dataset": "d"}},
    }


class _Proxy:
    def __init__(self) -> None:
        self.finished: list[str] = []

    def finish(self, trajectory: str) -> None:
        self.finished.append(trajectory)

    def turns(self, trajectory: str) -> list[dict]:
        return [{"trajectory": trajectory, "status": 200}]


def _agent(outcomes: dict):
    """A fake agent: ``outcomes[uid]`` is a list of results (or exceptions) handed out in order."""
    calls = []

    async def run(*, base_url, prompt, request_kwargs, metadata):
        calls.append({"base_url": base_url, "prompt": prompt, "request_kwargs": request_kwargs, "metadata": metadata})
        result = outcomes[metadata["uid"]].pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    return run, calls


def _scored(reward: float, category: str) -> dict:
    return {
        "reward": reward,
        "num_turns": 7,
        "stop_reason": "final_answer",
        "reward_details": {"category": category, "steps": 3},
        "agent_metrics": {"model_retries": 0},
    }


async def _run(sets, agent, proxy=None, samples=2):
    return await driver.run_sets(
        sets,
        agent_run=agent,
        proxy=proxy or _Proxy(),
        proxy_url="http://127.0.0.1:1234",
        samples_per_prompt=samples,
        max_concurrent=3,
        max_seq_len=150_000,
        attempt_id="a1",
    )


async def test_each_trajectory_gets_its_own_session_route_and_the_rollout_request() -> None:
    agent, calls = _agent({"q1": [_scored(1.0, "correct"), _scored(0.0, "wrong")]})
    proxy = _Proxy()

    await _run({"heldout": [_row("q1")]}, agent, proxy)

    sessions = {call["metadata"]["session_id"] for call in calls}
    routes = {call["base_url"] for call in calls}
    assert len(sessions) == 2 and len(routes) == 2  # observation ids are numbered per session
    assert all(route.startswith("http://127.0.0.1:1234/t/heldout-q1-") for route in routes)
    assert sorted(proxy.finished) == sorted(route.rsplit("/", 1)[1] for route in routes)
    call = calls[0]
    assert call["request_kwargs"] == driver.REQUEST_KWARGS and call["request_kwargs"] is not driver.REQUEST_KWARGS
    assert call["metadata"]["max_seq_len"] == 150_000 and call["metadata"]["attempt_id"] == "a1"
    assert call["metadata"]["tool_context"] == {"dataset": "d"} and call["prompt"][1]["content"] == "question q1"


async def test_rewards_categories_and_lost_trajectories_reduce_like_an_in_run_sweep() -> None:
    agent, _ = _agent(
        {
            "q1": [_scored(1.0, "correct"), _scored(0.0, "wrong")],
            "q2": [EnvironmentFailure("model transport failure"), _scored(0.0, "fabricated")],
        }
    )

    outcomes = await _run({"heldout": [_row("q1"), _row("q2")]}, agent)

    outcome = outcomes["heldout"]
    # Trajectories finish in any order, so compare each question's rewards as a multiset.
    assert {uid: sorted(rewards) for uid, rewards in outcome.rewarded.items()} == {"q1": [0.0, 1.0], "q2": [0.0]}
    assert outcome.lost == 1 and outcome.total == 4
    assert dict(outcome.categories) == {"correct": 1, "wrong": 1, "fabricated": 1}
    metrics = driver.set_metrics("heldout", outcome)
    assert metrics["officeqa_eval/heldout/pass_rate"] == pytest.approx(1 / 3)
    assert metrics["officeqa_eval/heldout/pass_rate_lost_as_zero"] == pytest.approx(1 / 4)
    assert metrics["officeqa_eval/heldout/questions_solved"] == 1
    assert metrics["officeqa_eval/heldout/lost_trajectories"] == 1
    lost = [record for record in outcome.trajectories if record["status"] == "lost"]
    assert len(lost) == 1 and lost[0]["error"].startswith("environment:") and lost[0]["turns"]


async def test_an_agent_bug_or_bad_reward_is_lost_not_scored() -> None:
    agent, _ = _agent({"q1": [RuntimeError("bug"), {**_scored(1.0, "correct"), "reward": float("nan")}]})

    outcome = (await _run({"heldout": [_row("q1")]}, agent))["heldout"]

    assert outcome.lost == 2 and dict(outcome.rewarded) == {}
    assert {record["error"].split(":")[0] for record in outcome.trajectories} == {"RuntimeError", "ValueError"}


async def test_an_unreported_trajectory_counts_as_no_report() -> None:
    result = {"reward": 0.0, "num_turns": 128, "stop_reason": "max_turns", "agent_metrics": {}}
    agent, _ = _agent({"q1": [result]})

    outcome = (await _run({"heldout": [_row("q1")]}, agent, samples=1))["heldout"]

    assert dict(outcome.categories) == {"no_report": 1}


def test_rows_must_carry_what_the_agent_needs(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps(_row("q1")) + "\n" + json.dumps(_row("q1")) + "\n")
    with pytest.raises(ValueError, match="unique"):
        driver.load_rows(path)
    broken = _row("q2")
    del broken["metadata"]["tool_context"]
    path.write_text(json.dumps(broken) + "\n")
    with pytest.raises(ValueError, match="tool_context"):
        driver.load_rows(path)


async def test_outputs_keep_option_a_schema_and_publish_once(tmp_path: Path) -> None:
    agent, _ = _agent({"q1": [_scored(1.0, "correct")]})
    outcomes = await _run({"heldout": [_row("q1")]}, agent, samples=1)
    results = driver.results_document(
        run_id="r", rollout_id=99, identity={"kind": "export"}, outcomes=outcomes, valid=True
    )

    driver.write_outputs(tmp_path / "eval", results=results, summary={"valid": True}, outcomes=outcomes, captured={})

    document = json.loads((tmp_path / "eval/eval_results.json").read_text())
    assert document["results"] == {"heldout": {"q1": [1.0]}} and document["harness"] == "hf_sglang_v1"
    assert (document["schema"], document["rollout_id"], document["valid"]) == (1, 99, True)
    assert len((tmp_path / "eval/trajectories.jsonl").read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        driver.write_outputs(tmp_path / "eval", results=results, summary={}, outcomes=outcomes, captured={})


def test_eval_modules_import_without_torch_ray_or_megatron() -> None:
    script = (
        "import sys\n"
        "for name in ('torch', 'ray', 'megatron', 'sglang', 'transformers'):\n"
        "    sys.modules[name] = None\n"
        "import examples.glm5_3_flash.officeqa.eval.driver, examples.glm5_3_flash.officeqa.eval.proxy, examples.glm5_3_flash.officeqa.eval.server\n"
        "import examples.glm5_3_flash.agent\n"
        "import examples.glm5_3_flash.officeqa.eval_summary\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[5])
