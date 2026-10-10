import json
import shlex

import pytest
from tests.fast.launch_scripts.py_harness import import_launch_script
from tests.fast.launch_scripts.sh_harness import REPO_ROOT
from typer.testing import CliRunner


@pytest.fixture
def launcher(monkeypatch):
    module = import_launch_script(REPO_ROOT / "scripts/run_glm5_3_flash.py")
    for key in ("SLURM_JOB_NUM_NODES", "DATABRICKS_HOST", "DATABRICKS_TOKEN", "MLFLOW_RUN_ID", "MILES_ATTEMPT_ID"):
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


def test_officeqa_on_eight_nodes_is_the_default_recipe(launcher):
    call, tokens = _run(launcher)

    assert call["train_script"] == "train_async.py"
    assert call["megatron_model_type"] == "glm5.3-flash"
    # Four nodes train with CP8 at 150K tokens; the other four serve as 8-GPU SGLang engines.
    assert _value(tokens, "--actor-num-nodes") == "4"
    assert _value(tokens, "--rollout-num-gpus") == "32"
    assert _value(tokens, "--rollout-num-gpus-per-engine") == "8"
    assert _value(tokens, "--context-parallel-size") == "8"
    assert _value(tokens, "--seq-length") == "150000"
    assert _value(tokens, "--rollout-max-response-len") == "16384"
    assert "--fully-async" in tokens and "--colocate" not in tokens
    assert _value(tokens, "--prompt-data") == "/root/datasets/officeqa/train.jsonl"
    assert _value(tokens, "--data-source-path") == "examples.glm5_3_flash.officeqa.curriculum.CurriculumDataSource"
    assert _value(tokens, "--custom-agent-function-path") == "examples.glm5_3_flash.agent.run"
    assert "--custom-config-path" not in tokens
    assert "--session-capture-provenance" in tokens and "--use-rollout-routing-replay" in tokens
    assert "--disable-grpo-std-normalization" in tokens and "--sglang-disable-radix-cache" not in tokens
    assert "--rm-type" not in tokens


def test_officeqa_runs_need_no_environment_setup(launcher):
    call, _ = _run(launcher, data_dir="/data")

    env = call["extra_env_vars"]
    assert env["MILES_TOOL_EXECUTOR"] == "examples.glm5_3_flash.officeqa.adapters:run_tool"
    assert env["MILES_REWARD_FN"] == "examples.glm5_3_flash.officeqa.path_reward:score"
    assert json.loads(env["MILES_AGENT_SETTINGS"])["terminal_tools"] == ["submit_report"]
    assert env["OFFICEQA_DATA_ROOT"] == "/data"


def test_judge_credentials_reach_the_workers_when_set(launcher, monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://example.cloud.databricks.com")
    monkeypatch.setenv("DATABRICKS_TOKEN", "token")

    call, _ = _run(launcher)

    assert call["extra_env_vars"]["DATABRICKS_HOST"] == "https://example.cloud.databricks.com"
    assert call["extra_env_vars"]["DATABRICKS_TOKEN"] == "token"


def test_generic_extra_args_and_environment_reach_the_driver_and_workers(launcher):
    module, _ = launcher
    call, tokens = _run(
        launcher,
        extra_args="--checkpoint-publisher-path example.publish",
        extra_env_vars=json.dumps({"CHECKPOINT_CONFIG": "/local/runtime.json"}),
    )
    env = module.U.resolve_extra_env_vars(call["extra_env_vars"], call["config"])
    assert env["CHECKPOINT_CONFIG"] == "/local/runtime.json"
    assert _value(tokens, "--checkpoint-publisher-path") == "example.publish"


@pytest.mark.parametrize("skip_download", [False, True])
def test_prepare_converts_a_staged_checkpoint_without_downloading_it(launcher, monkeypatch, skip_download):
    module, _ = launcher
    commands, conversions = [], []
    monkeypatch.setattr(module.U, "exec_command_cpu", commands.append)
    monkeypatch.setattr(module.U, "convert_checkpoint", lambda **kwargs: conversions.append(kwargs))

    module._prepare(module.ScriptArgs(run_id="test-run", model_dir="/staged/models", skip_download=skip_download))

    assert any(command.startswith("hf download zai-org/GLM-5.3-Flash") for command in commands) is not skip_download
    assert [call["hf_checkpoint"] for call in conversions] == ["/staged/models/GLM-5.3-Flash"]


def test_more_nodes_add_rollout_engines(launcher):
    _, tokens = _run(launcher, num_nodes=12)
    assert _value(tokens, "--actor-num-nodes") == "4"
    assert _value(tokens, "--rollout-num-gpus") == "64"


def test_save_hf_exports_inside_each_save_in_the_base_layout(launcher):
    _, tokens = _run(launcher, save_hf=True, save_interval=25)

    hf_path = _value(tokens, "--save-hf")
    assert hf_path.format(rollout_id=99).endswith("/checkpoints/iter_0000099/hf")
    assert hf_path.startswith(_value(tokens, "--save") + "/")
    assert _value(tokens, "--save-interval") == "25"
    assert _value(tokens, "--save-hf-layout-path") == "examples.glm5_3_flash.hf_export.base_layout"


def test_a_persist_dir_is_where_saves_are_copied_and_resumed_from(launcher, tmp_path):
    persist = tmp_path / "checkpoints"
    _, fresh = _run(launcher, persist_dir=str(persist))
    (persist / "latest_checkpointed_iteration.txt").parent.mkdir()
    (persist / "latest_checkpointed_iteration.txt").write_text("9")
    launcher[1].clear()
    _, resumed = _run(launcher, persist_dir=str(persist))

    for tokens in (fresh, resumed):
        assert _value(tokens, "--load") == str(persist)
        assert _value(tokens, "--checkpoint-publisher-path") == "tools.air.publisher.VolumePublisher"
    # Only a fresh start checks the initial weight sync against the reference.
    assert "--check-weight-update-equal" in fresh and "--check-weight-update-equal" not in resumed


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_nodes": 4}, "rollout engines"),
        ({"num_gpus_per_node": 4, "num_nodes": 16}, "eight-GPU nodes"),
        ({"save_hf": True, "skip_saving": True}, "inside a save"),
        ({"n_samples_per_prompt": 1}, "two samples"),
        ({"rollout_max_response_len": 150000}, "room for a prompt"),
    ],
)
def test_misconfigured_runs_are_rejected(launcher, overrides, match):
    with pytest.raises(ValueError, match=match):
        launcher[0].ScriptArgs(run_id="test-run", **overrides)


def test_token_capacity_rounds_up_to_the_local_quantum(launcher):
    module, _ = launcher
    assert module._max_tokens_per_gpu(150000) == 19456


@pytest.mark.parametrize("command", ["train", "prepare"])
def test_cli_accepts_the_documented_subcommands(launcher, command):
    module, _ = launcher
    result = CliRunner().invoke(module.app, [command, "--help"])
    assert result.exit_code == 0, result.output
