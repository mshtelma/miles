"""The standard GLM-5.3-Flash launcher: fully async DAPO on DAPO-Math-17k, any other task by flags."""

import shlex

import pytest
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT


@pytest.fixture
def launcher(monkeypatch):
    module = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash.py")
    for key in ("SLURM_JOB_NUM_NODES", "MLFLOW_RUN_ID", "MILES_ATTEMPT_ID"):
        monkeypatch.delenv(key, raising=False)
    calls = []
    monkeypatch.setattr(module.U, "execute_train", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(module.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    return module, calls


def _run(launcher, **overrides):
    module, calls = launcher
    module._train(module.ScriptArgs(run_id="test-run", **overrides))
    assert len(calls) == 1
    return calls[0], shlex.split(calls[0]["train_args"])


def _value(tokens, flag):
    return tokens[tokens.index(flag) + 1]


def test_the_default_is_async_math_on_eight_nodes(launcher):
    call, tokens = _run(launcher)

    assert call["train_script"] == "train_async.py"
    assert call["megatron_model_type"] == "glm5.3-flash"
    # Four nodes train with TP1, CP8, PP4 and EP8; the other four serve as 8-GPU engines.
    assert _value(tokens, "--actor-num-nodes") == "4"
    assert _value(tokens, "--rollout-num-gpus") == "32"
    assert [_value(tokens, f"--{name}-parallel-size") for name in ("tensor-model", "context", "pipeline-model")] == [
        "1",
        "8",
        "4",
    ]
    assert _value(tokens, "--prompt-data") == "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl"
    assert _value(tokens, "--rm-type") == "deepscaler" and "--apply-chat-template" in tokens
    assert _value(tokens, "--eval-prompt-data") == "aime24"
    # Without the session server, routes survive a weight update only when requests pause in place.
    assert _value(tokens, "--pause-generation-mode") == "in_place" and "--use-rollout-routing-replay" in tokens
    assert "--use-session-server" not in tokens and "--custom-agent-function-path" not in tokens


def test_another_task_is_a_matter_of_flags(launcher):
    _, tokens = _run(launcher, prompt_data="/data/task.jsonl", enable_eval=False, extra_args="--custom-rm-path my.rm")

    assert _value(tokens, "--prompt-data") == "/data/task.jsonl"
    assert "--eval-prompt-data" not in tokens
    assert _value(tokens, "--custom-rm-path") == "my.rm"


def test_save_hf_exports_in_the_base_checkpoints_layout(launcher):
    _, tokens = _run(launcher, save_hf=True)

    assert _value(tokens, "--save-hf").endswith("/checkpoints/iter_{rollout_id:07d}/hf")
    assert _value(tokens, "--save-hf-layout-path") == "miles.backends.megatron_utils.hf_base_layout.base_layout"


def test_prepare_downloads_the_pinned_model_and_the_datasets(launcher, monkeypatch):
    module, _ = launcher
    commands, datasets, conversions = [], [], []
    monkeypatch.setattr(module.U, "exec_command_cpu", commands.append)
    monkeypatch.setattr(module.U, "hf_download_dataset", lambda name, data_dir: datasets.append(name))
    monkeypatch.setattr(module.U, "convert_checkpoint", lambda **kwargs: conversions.append(kwargs))

    module._prepare(module.ScriptArgs(run_id="test-run"))

    assert any(
        command.startswith(f"hf download zai-org/GLM-5.3-Flash --revision {module._HF_REVISION}")
        for command in commands
    )
    assert datasets == ["zhuzilin/dapo-math-17k", "zhuzilin/aime-2024"]
    [conversion] = conversions
    assert conversion["dir_dst"] == "/root/ckpt" and "--pipeline-model-parallel-size 4" in conversion["extra_args"]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_nodes": 4}, "rollout engines"),
        ({"num_gpus_per_node": 4, "num_nodes": 16}, "eight-GPU nodes"),
        ({"rollout_max_response_len": 32768}, "room for a prompt"),
    ],
)
def test_misconfigured_runs_are_rejected(launcher, overrides, match):
    with pytest.raises(ValueError, match=match):
        launcher[0].ScriptArgs(run_id="test-run", **overrides)
