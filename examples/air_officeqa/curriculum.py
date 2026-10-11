"""Adaptive question sampling for OfficeQA RL on the fully-async rollout.

``CurriculumDataSource`` is the GLM-5.3 launcher's ``--data-source-path``. Its settings live in
``curriculum.json`` beside the prompt file (prepare_data.py writes it):

    seed                     sampling seed
    floor                    least weight a question keeps
    decay                    factor on a question's counts each time a group of it finishes
    max_groups_per_question  groups of one question in flight at once

Questions are drawn with weight p(1 - p), p being the question's estimated solve rate, so groups that
are all right or all wrong, which carry no GRPO advantage, are rarely generated. The weight never drops
below the floor: a question that is always or never solved still comes back now and then, and is
picked up again if it becomes learnable.

A group whose judge left a verdict unresolved is generated again, at most twice: an unresolved verdict
is not a wrong answer, and a zero would skew the group's baseline. After that the group is dropped.
"""

import collections
import copy
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

from examples.air_officeqa.eval_summary import outcome_from_metadata
from miles.rollout import data_source
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

CONFIG_FILE = "curriculum.json"
_UNJUDGED_RETRIES = 2


@dataclass
class _Question:
    rewarded: float = 0.0
    trials: float = 0.0
    in_flight: set[int] = field(default_factory=set)

    @property
    def solve_rate(self) -> float:
        return (self.rewarded + 1) / (self.trials + 2)  # Laplace: an unseen question starts at 0.5


class Curriculum:
    """Which question to generate next, and the per-question solve rates and metrics behind it."""

    def __init__(self, settings: dict, uids: list[str]):
        self._floor = float(settings["floor"])
        self._decay = float(settings["decay"])
        self._cap = int(settings["max_groups_per_question"])
        if not 0 < self._floor <= 0.25 or not 0 <= self._decay <= 1 or self._cap <= 0:
            raise ValueError("curriculum floor, decay or group cap is out of range")
        if not uids:
            raise ValueError("the curriculum needs training questions")
        self._rng = random.Random(settings["seed"])
        self._questions = {uid: _Question() for uid in uids}
        self._step: list[dict] = []
        self._step_groups = self._step_mixed = self._requeued = self._dropped = 0

    def next_question(self) -> str:
        eligible = [uid for uid, question in self._questions.items() if len(question.in_flight) < self._cap]
        uids = eligible or list(self._questions)
        return self._rng.choices(uids, weights=[self.weight(uid) for uid in uids])[0]

    def weight(self, uid: str) -> float:
        rate = self._questions[uid].solve_rate
        return max(rate * (1 - rate), self._floor)

    def issued(self, uid: str, group_id: int) -> None:
        self._questions[uid].in_flight.add(group_id)

    def finished(self, uid: str, group_id: int, outcomes: list[dict]) -> None:
        question = self._questions[uid]
        question.in_flight.discard(group_id)
        rewarded = sum(outcome["reward"] > 0 for outcome in outcomes)
        question.rewarded = self._decay * question.rewarded + rewarded
        question.trials = self._decay * question.trials + len(outcomes)
        self._step.extend(outcomes)
        self._step_groups += 1
        self._step_mixed += 0 < rewarded < len(outcomes)

    def requeued(self) -> None:
        self._requeued += 1

    def dropped(self, uid: str, group_id: int) -> None:
        self._questions[uid].in_flight.discard(group_id)
        self._dropped += 1

    def metrics(self) -> dict[str, float]:
        """Metrics of the groups finished since the last call."""
        rates = [question.solve_rate for question in self._questions.values()]
        metrics = {
            "curriculum/groups": self._step_groups,
            "curriculum/requeued_unjudged_groups": self._requeued,
            "curriculum/dropped_unjudged_groups": self._dropped,
            "curriculum/learnable_questions": sum(0.1 <= rate <= 0.9 for rate in rates),
        }
        if self._step_groups:
            metrics["curriculum/mixed_group_share"] = self._step_mixed / self._step_groups
        if outcomes := self._step:
            metrics["curriculum/solve_rate"] = sum(outcome["reward"] > 0 for outcome in outcomes) / len(outcomes)
            for category, count in collections.Counter(outcome["category"] for outcome in outcomes).items():
                metrics[f"report/{category}"] = count / len(outcomes)
            reports = [outcome for outcome in outcomes if outcome["steps"]]
            if reports:
                metrics["report/steps"] = sum(outcome["steps"] for outcome in reports) / len(reports)
                metrics["report/late_input_share"] = sum(outcome["late_inputs"] > 0 for outcome in reports) / len(
                    reports
                )
            if compute := sum(outcome["compute_steps"] for outcome in outcomes):
                metrics["report/declared_input_share"] = (
                    sum(outcome["compute_steps_with_inputs"] for outcome in outcomes) / compute
                )
            metrics["report/turns"] = sum(outcome["turns"] for outcome in outcomes) / len(outcomes)
        self._step = []
        self._step_groups = self._step_mixed = self._requeued = self._dropped = 0
        return metrics

    def state(self) -> dict:
        version, internal, gauss = self._rng.getstate()
        return {
            "questions": {uid: [q.rewarded, q.trials] for uid, q in self._questions.items()},
            "rng": [version, list(internal), gauss],
        }

    def restore(self, state: dict) -> None:
        for uid, (rewarded, trials) in state["questions"].items():
            if uid in self._questions:
                self._questions[uid].rewarded, self._questions[uid].trials = rewarded, trials
        version, internal, gauss = state["rng"]
        self._rng.setstate((version, tuple(internal), gauss))


class CurriculumDataSource(data_source.RolloutDataSourceWithBuffer):
    """The built-in buffered source, drawing prompts from a ``Curriculum``.

    The fully-async rollout calls ``on_group_finished`` with every finished group before it reaches the
    training buffer, and logs ``rollout_metrics`` with each training step.
    """

    def __init__(self, args):
        super().__init__(args)
        if self.dataset is None:
            raise ValueError("the curriculum samples from the global prompt dataset")
        settings = json.loads((Path(args.prompt_data).parent / CONFIG_FILE).read_text(encoding="utf-8"))
        self._prompts = _by_uid(self.dataset.samples)
        self.curriculum = Curriculum(settings, list(self._prompts))
        self._issued: dict[int, str] = {}  # group id -> uid, for the groups in flight
        self._unjudged_retries: dict[int, int] = {}

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        while len(samples) < num_samples:
            uid = self.curriculum.next_question()
            group_id = self.sample_group_index
            self.sample_group_index += 1
            self.sample_index = self.sample_group_index * self.args.n_samples_per_prompt
            self._issued[group_id] = uid
            self.curriculum.issued(uid, group_id)
            samples.append(self._group(group_id))
        return samples

    def _group(self, group_id: int) -> list[Sample]:
        group = []
        for member in range(self.args.n_samples_per_prompt):
            sample = copy.deepcopy(self._prompts[self._issued[group_id]])
            sample.group_index = group_id
            sample.index = group_id * self.args.n_samples_per_prompt + member
            group.append(sample)
        return group

    def on_group_finished(self, group: list[Sample]) -> bool:
        """Record a finished group; True keeps it out of training (to judge it again, or dropped)."""
        group_id = group[0].group_index
        uid = self._issued.get(group_id)
        if uid is None:
            return False
        if any(sample.status == Sample.Status.ABORTED for sample in group):
            return False  # the rollout retries a training group as usual
        outcomes = [_outcome(sample) for sample in group]
        if any(outcome["category"] == "unknown" for outcome in outcomes):
            retries = self._unjudged_retries.get(group_id, 0)
            if retries < _UNJUDGED_RETRIES:
                self._unjudged_retries[group_id] = retries + 1
                self.curriculum.requeued()
                self.buffer.append(self._group(group_id))
                return True
            logger.warning("curriculum: dropping group %d of %s, unjudged after %d retries", group_id, uid, retries)
            self._release(group_id)
            self.curriculum.dropped(uid, group_id)
            return True
        self._release(group_id)
        self.curriculum.finished(uid, group_id, outcomes)
        return False

    def _release(self, group_id: int) -> None:
        del self._issued[group_id]
        self._unjudged_retries.pop(group_id, None)

    def rollout_metrics(self) -> dict[str, float]:
        return self.curriculum.metrics()

    def save(self, rollout_id):
        # The base source saves and restores ``metadata`` with its cursor. Groups in flight are not
        # saved: they die with the process, and a resumed run issues new ones.
        self.metadata["curriculum"] = self.curriculum.state()
        super().save(rollout_id)

    def load(self, rollout_id=None):
        super().load(rollout_id)
        if (state := self.metadata.get("curriculum")) is not None:
            self.curriculum.restore(state)


def _by_uid(samples: list[Sample]) -> dict[str, Sample]:
    prompts = {}
    for sample in samples:
        uid = (sample.metadata or {}).get("uid")
        if not isinstance(uid, str) or uid in prompts:
            raise ValueError(f"curriculum prompts need unique string uids, got {uid!r}")
        prompts[uid] = sample
    return prompts


def _outcome(sample: Sample) -> dict:
    """What the curriculum and its metrics need from one finished trajectory."""
    metadata = sample.metadata or {}
    raw_reward = float(metadata.get("raw_reward", sample.reward or 0.0))
    return outcome_from_metadata(metadata, raw_reward)
