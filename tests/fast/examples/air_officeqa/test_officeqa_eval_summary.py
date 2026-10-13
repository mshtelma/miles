"""OfficeQA eval metrics: pass rates and pass@k of an eval set."""

import collections

import pytest

from examples.air_officeqa import eval_summary


def test_summary_reports_pass_rates_and_unbiased_pass_at_k():
    rewarded = {"q1": [1.0, 1.0, 0.0], "q2": [0.0, 0.0]}
    categories = collections.Counter({"correct": 2, "wrong": 3})

    metrics = eval_summary.summarize_eval_set("heldout", rewarded, categories, lost=0)

    assert metrics["officeqa_eval/heldout/pass_rate"] == 2 / 5
    assert metrics["officeqa_eval/heldout/questions_solved"] == 1
    # pass@1 macro = mean(c/n): q1=2/3, q2=0/2 -> 1/3. pass@3 only estimable for q1 (n>=3): 1 - C(1,3)/C(3,3) = 1.
    assert metrics["officeqa_eval/heldout/pass_at_1"] == pytest.approx(1 / 3)
    assert metrics["officeqa_eval/heldout/pass_at_3"] == 1.0


def test_pass_at_k_is_the_unbiased_estimator_and_skips_undersampled_questions():
    # c=8/8 always passes; c=4/8 is the Chen et al. mid case; c=0/8 never passes; n=2<3 can't estimate pass@3.
    rewarded = {"all": [1.0] * 8, "half": [1.0] * 4 + [0.0] * 4, "none": [0.0] * 8, "short": [1.0, 0.0]}

    assert eval_summary.pass_at_k(rewarded, 1) == pytest.approx((1.0 + 0.5 + 0.0 + 0.5) / 4)  # pass@1 == mean(c/n)
    # pass@3 skips "short" (n=2<3): all->1, half->1-C(4,3)/C(8,3)=1-4/56, none->1-C(8,3)/C(8,3)=0.
    assert eval_summary.pass_at_k(rewarded, 3) == pytest.approx((1.0 + (1.0 - 4 / 56) + 0.0) / 3)
    assert eval_summary.pass_at_k({}, 3) == 0.0  # no questions -> defined as 0, never divide by zero


def test_lost_trajectories_are_excluded_from_the_rate_but_counted():
    rewarded = {"q1": [1.0, 0.0]}

    metrics = eval_summary.summarize_eval_set("heldout", rewarded, collections.Counter({"correct": 1}), lost=1)

    assert metrics["officeqa_eval/heldout/pass_rate"] == 0.5  # 1 of the 2 scored trajectories passed
    assert metrics["officeqa_eval/heldout/lost_trajectories"] == 1


def test_pass_at_k_is_left_out_when_no_question_has_k_samples():
    metrics = eval_summary.summarize_eval_set("heldout", {"q1": [1.0, 0.0]}, collections.Counter(), lost=0)

    assert metrics["officeqa_eval/heldout/pass_at_1"] == 0.5
    assert "officeqa_eval/heldout/pass_at_3" not in metrics  # 2 samples can't estimate pass@3
