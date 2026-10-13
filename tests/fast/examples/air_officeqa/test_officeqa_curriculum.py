"""OfficeQA curriculum: p(1 - p) sampling and the data-source wiring."""

import collections
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.air_officeqa import curriculum
from miles.utils.types import Sample

_SETTINGS = {"seed": 0, "floor": 0.01, "decay": 0.75, "max_groups_per_question": 2}


def _outcome(reward: float, category: str = "correct", **fields) -> dict:
    defaults = {"steps": 3, "compute_steps": 1, "compute_steps_with_inputs": 1, "late_inputs": 0, "turns": 9}
    return {"reward": reward, "category": category, **defaults, **fields}


def _trained(plan: curriculum.Curriculum, uid: str, rewarded: int, trials: int) -> None:
    plan.finished(uid, 0, [_outcome(1.0)] * rewarded + [_outcome(0.0, "wrong")] * (trials - rewarded))


def test_questions_the_policy_sometimes_solves_are_drawn_most():
    plan = curriculum.Curriculum({**_SETTINGS, "decay": 1.0}, ["mixed", "solved", "unsolved"])
    _trained(plan, "mixed", 4, 8)
    _trained(plan, "solved", 40, 40)
    _trained(plan, "unsolved", 0, 40)

    drawn = collections.Counter(plan.next_question() for _ in range(3000))

    # p(1 - p) is about 0.25 at a 50% solve rate and 0.023 at 40 of 40 trials; neither extreme vanishes.
    assert drawn["mixed"] > 5 * drawn["solved"] > 0 and drawn["mixed"] > 5 * drawn["unsolved"] > 0


def test_a_question_at_its_in_flight_cap_is_not_drawn():
    plan = curriculum.Curriculum(_SETTINGS, ["mixed", "solved"])
    plan.issued("mixed", 0)
    plan.issued("mixed", 1)

    assert {plan.next_question() for _ in range(50)} == {"solved"}

    plan.finished("mixed", 0, [_outcome(1.0)] * 8)
    assert "mixed" in {plan.next_question() for _ in range(50)}


def test_finished_groups_move_the_estimate_and_older_counts_decay():
    plan = curriculum.Curriculum(_SETTINGS, ["q"])
    _trained(plan, "q", 0, 8)
    _trained(plan, "q", 8, 8)

    # 8 trials, none rewarded, decay to 6 and 0; then 8 of 8: (8 + 1) / (14 + 2)
    assert plan._questions["q"].solve_rate == pytest.approx(9 / 16)


def test_step_metrics_count_report_outcomes_and_declared_inputs():
    plan = curriculum.Curriculum(_SETTINGS, ["mixed"])
    outcomes = [_outcome(1.0)] * 2 + [_outcome(0.0, "wrong", compute_steps_with_inputs=0, late_inputs=1)] * 2
    plan.finished("mixed", 0, outcomes)

    metrics = plan.metrics()

    assert (metrics["curriculum/groups"], metrics["curriculum/mixed_group_share"]) == (1, 1.0)
    assert (metrics["report/correct"], metrics["report/wrong"]) == (0.5, 0.5)
    assert metrics["report/declared_input_share"] == 0.5
    assert metrics["report/late_input_share"] == 0.5
    assert plan.metrics()["curriculum/groups"] == 0


def test_state_round_trips_through_a_checkpoint():
    plan = curriculum.Curriculum(_SETTINGS, ["mixed", "solved"])
    plan.next_question()
    plan.finished("mixed", 0, [_outcome(1.0)] * 8)

    restored = curriculum.Curriculum(_SETTINGS, ["mixed", "solved"])
    restored.restore(json.loads(json.dumps(plan.state())))

    assert [restored.next_question() for _ in range(6)] == [plan.next_question() for _ in range(6)]


# ---------------------------------------------------------------- data source


class _Prompts:
    def __init__(self, path, **kwargs):
        rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
        self.samples = [Sample(prompt=row["prompt"], metadata=row["metadata"]) for row in rows]

    def __len__(self):
        return len(self.samples)

    def shuffle(self, epoch):
        pass


def _source(tmp_path, monkeypatch, *, settings) -> curriculum.CurriculumDataSource:
    rows = [{"prompt": f"question {uid}", "metadata": {"uid": uid}} for uid in ("mixed", "solved")]
    (tmp_path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (tmp_path / curriculum.CONFIG_FILE).write_text(json.dumps(settings))
    monkeypatch.setattr("miles.rollout.data_source.load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr("miles.rollout.data_source.load_processor", lambda *args, **kwargs: object())
    monkeypatch.setattr("miles.rollout.data_source.Dataset", _Prompts)
    args = SimpleNamespace(
        rollout_global_dataset=True,
        rollout_shuffle=False,
        fully_async=True,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data=str(tmp_path / "train.jsonl"),
        rollout_max_prompt_len=32,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs={},
        rollout_seed=0,
        n_samples_per_prompt=2,
        buffer_filter_path=None,
        save=str(tmp_path / "ckpt"),
        load=None,
    )
    return curriculum.CurriculumDataSource(args)


def _finish(group, rewards, **metadata):
    for sample, reward in zip(group, rewards, strict=True):
        sample.status = Sample.Status.COMPLETED
        sample.reward = reward
        sample.metadata = {**sample.metadata, "raw_reward": reward, "stop_reason": "final_answer", **metadata}
    return group


def test_a_finished_group_is_trained_on_and_released(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, settings=_SETTINGS)
    [group] = source.get_samples(1)

    assert source.on_group_finished(_finish(group, [1.0, 0.0])) is False
    assert source.rollout_metrics()["curriculum/groups"] == 1
    assert source._issued == {}


def test_an_unjudged_group_is_generated_again_at_most_twice_then_dropped(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, settings=_SETTINGS)
    [group] = source.get_samples(1)
    group_id = group[0].group_index

    for _ in range(2):
        assert source.on_group_finished(_finish(group, [0.0, 0.0], reward_details={"category": "unknown"})) is True
        [group] = source.get_samples(1)
        assert [sample.group_index for sample in group] == [group_id] * 2

    assert source.on_group_finished(_finish(group, [0.0, 0.0], reward_details={"category": "unknown"})) is True
    metrics = source.rollout_metrics()
    assert (metrics["curriculum/requeued_unjudged_groups"], metrics["curriculum/dropped_unjudged_groups"]) == (2, 1)
    assert source._issued == {} and source._unjudged_retries == {}
    assert source.get_samples(1)[0][0].group_index != group_id  # a new group, not the dropped one


def test_an_aborted_group_is_left_to_the_rollout_retry(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, settings=_SETTINGS)
    [group] = source.get_samples(1)
    group[0].status = Sample.Status.ABORTED

    assert source.on_group_finished(group) is False
    assert source.rollout_metrics()["curriculum/groups"] == 0


def test_the_source_resumes_its_curriculum_from_a_checkpoint(tmp_path, monkeypatch):
    source = _source(tmp_path, monkeypatch, settings=_SETTINGS)
    groups = source.get_samples(3)
    source.on_group_finished(_finish(groups[0], [1.0, 0.0]))
    source.save(4)

    restored = _source(tmp_path, monkeypatch, settings=_SETTINGS)
    restored.args.load = restored.args.save
    restored.load(4)

    assert restored.sample_group_index == source.sample_group_index == 3
    assert [restored.curriculum.next_question() for _ in range(6)] == [
        source.curriculum.next_question() for _ in range(6)
    ]
