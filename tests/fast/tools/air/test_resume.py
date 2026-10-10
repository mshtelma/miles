import copy

import pytest
from tests.fast.tools.air.conftest import copy_save

from tools.air import publisher, resume


def _config(inputs, reference, argv=()):
    return resume.configuration(
        [
            "--ckpt-format",
            "torch_dist",
            "--hf-checkpoint",
            "/local/models/GLM-5.3-Flash",
            "--num-rollout",
            "100",
            "--global-batch-size",
            "8",
            "--rollout-batch-size",
            "4",
            "--n-samples-per-prompt",
            "2",
            "--lr",
            "1e-6",
            "--tito-model",
            "glm53",
            *argv,
        ],
        inputs[1],
        "GLM-5.3-Flash",
        inputs[0].stem,
        reference[1],
    )


def _publish(checkpoint, configuration):
    return publisher.finalize_save(
        checkpoint.target,
        copy_save(checkpoint),
        run_id="run",
        attempt_id=checkpoint.attempt,
        rollout_id=checkpoint.step,
        configuration=configuration,
    )


def test_latest_ignores_partial_and_missing_shard_saves(distributed_save, input_manifest, reference_root):
    configuration = _config(input_manifest, reference_root)
    old = distributed_save(2, attempt="old")
    _publish(old, configuration)
    partial = distributed_save(4, attempt="partial")
    copy_save(partial)
    bad = distributed_save(5, attempt="damaged")
    _publish(bad, configuration)
    next(bad.target.rglob("*.distcp")).unlink()
    selected = resume.latest_complete(old.target.parents[4], "run")
    assert selected[0] == old.target
    assert selected[1]["rollout_id"] == 2


def test_no_saves_means_fresh_start(tmp_path):
    assert resume.latest_complete(tmp_path, "fresh") is None


def test_lr_horizon_error_names_saved_and_requested_values(input_manifest, reference_root):
    previous = _config(input_manifest, reference_root)
    requested = _config(input_manifest, reference_root, ["--num-rollout", "200"])
    with pytest.raises(ValueError, match="saved num_rollout=100.*requested num_rollout=200"):
        resume.check_compatibility({"configuration": previous}, requested)
    assert previous["lr_schedule"]["lr_decay_steps"] == 800
    assert requested["lr_schedule"]["wd_incr_steps"] == 1600


@pytest.mark.parametrize("section", ["model", "tokenizer", "parallel_layout", "optimizer", "inputs"])
def test_each_resume_fingerprint_is_checked(input_manifest, reference_root, section):
    previous = _config(input_manifest, reference_root)
    requested = copy.deepcopy(previous)
    requested[section]["changed"] = True
    with pytest.raises(ValueError, match=section):
        resume.check_compatibility({"configuration": previous}, requested)
    assert resume.fingerprints(previous)[section] != resume.fingerprints(requested)[section]


def test_explicit_lr_horizon_and_warmup_are_recorded(input_manifest, reference_root):
    config = _config(input_manifest, reference_root, ["--lr-decay-iters", "80", "--lr-warmup-fraction", "0.1"])
    assert config["lr_schedule"]["lr_decay_steps"] == 640
    assert config["lr_schedule"]["lr-warmup-fraction"] == ["0.1"]


def test_template_contents_and_registered_kwargs_are_fingerprinted(input_manifest, reference_root, tmp_path):
    template = tmp_path / "custom.jinja"
    template.write_text("first")
    arguments = ["--tito-model", "default", "--chat-template-path", str(template)]
    previous = _config(input_manifest, reference_root, arguments)
    template.write_text("second")
    requested = _config(input_manifest, reference_root, arguments)
    with pytest.raises(ValueError, match="tokenizer"):
        resume.check_compatibility({"configuration": previous}, requested)
    glm = _config(input_manifest, reference_root)
    assert glm["tokenizer"]["renderer"]["kwargs"] == {"enable_thinking": True, "clear_thinking": False}


@pytest.mark.parametrize(
    "flag",
    [
        "--async-save",
        "--no-save-optim",
        "--no-load-rng",
        "--reset-rollout-data-source",
        "--ckpt-step",
        "--debug-disable-optimizer",
        "--override-opt-param-scheduler",
        "--override-ckpt-iteration",
    ],
)
def test_flags_that_break_full_state_resume_fail_preflight(input_manifest, reference_root, flag):
    with pytest.raises(ValueError, match="incompatible flags"):
        _config(input_manifest, reference_root, [flag])
