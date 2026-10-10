import json
import threading
import time

import pytest
from examples.glm5_3_flash.officeqa.air import recipe
from scripts import run_glm5_3_flash as launcher
from tests.fast.tools.air.conftest import copy_save
from tools.air import node_entry, process, publisher, resume, stage_models


@pytest.fixture
def node_args(tmp_path, monkeypatch):
    for key, value in {
        "NUM_NODES": "8",
        "LOCAL_WORLD_SIZE": "8",
        "NODE_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MLFLOW_RUN_ID": "attempt",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(launcher.U, "get_default_wandb_args", lambda *args, **kwargs: "")
    return node_entry.parse_args(
        [
            "--volume-root",
            str(tmp_path / "volume"),
            "--local-root",
            str(tmp_path / "scratch"),
            "--run-id",
            "run",
            "--launcher-path",
            "examples.glm5_3_flash.officeqa.air.recipe.make_launch",
        ]
    )[0]


def test_node_builds_recipe_argv_and_generic_environment(node_args, input_manifest, reference_root):
    launch, config = node_entry.prepare_launch(node_args, ["--save-hf", "--mlflow-experiment", "officeqa"])
    flags = resume.options(launch.training_args)
    assert flags["hf-checkpoint"] == [str(node_args.input_root / "models/GLM-5.3-Flash")]
    assert flags["ref-load"] == [config["sources"][0]["local_root"]]
    assert "load" not in flags
    assert flags["save"] == [config["checkpoint_root"]]
    assert flags["custom-megatron-init-path"] == ["tools.air.dcp_stage.initialize"]
    assert flags["checkpoint-publisher-path"] == ["tools.air.publisher.CheckpointPublisher"]
    assert flags["mlflow-run-id"] == ["attempt"]
    env = json.loads(launch.command[launch.command.index("--extra-env-vars") + 1])
    assert env["MILES_CHECKPOINT_CONFIG"] == str(node_args.state_root / "runtime.json")
    assert env["MILES_ATTEMPT_ID"] == "attempt"
    assert not any("MILES_AIR" in key for key in env)
    assert node_entry.launcher_env()["MILES_SCRIPT_EXTERNAL_RAY"] == "1"


def test_node_stages_full_weights_on_trainer_and_engine_nodes(node_args, input_manifest, reference_root):
    _, config = node_entry.prepare_launch(node_args, [])
    node_entry.stage_node(config)
    assert (node_args.input_root / "models/GLM-5.3-Flash/model.safetensors").read_bytes() == b"HF weights"
    assert (node_args.input_root / "data/officeqa/train.jsonl").is_file()
    assert stage_models.read_json(node_args.state_root / "runtime.json") == config


def test_resume_uses_latest_complete_and_fails_horizon_before_ray(
    node_args, input_manifest, reference_root, distributed_save, monkeypatch
):
    _, config = node_entry.prepare_launch(node_args, [])
    checkpoint = distributed_save()
    publisher.finalize_save(
        checkpoint.target,
        copy_save(checkpoint),
        run_id="run",
        attempt_id="attempt",
        rollout_id=3,
        configuration=config["configuration"],
    )
    node_args.attempt_id = "next-attempt"
    launch, resumed = node_entry.prepare_launch(node_args, [])
    assert resume.options(launch.training_args)["load"] == [str(node_args.state_root / "dcp/resume")]
    assert resumed["sources"][1]["remote_root"] == str(checkpoint.target)

    def no_ray(*args, **kwargs):
        pytest.fail("incompatible resume started Ray")

    monkeypatch.setattr(node_entry.subprocess, "run", no_ray)
    with pytest.raises(ValueError, match="--num-rollout / LR schedule horizon"):
        node_entry.run_head(node_args, ["--num-rollout", "200"], node_args.volume_root / "results")


@pytest.mark.parametrize(
    "argv",
    [
        ["--model-dir", "/other"],
        ["--resume-root=/other"],
        ["--extra-args", "--load /other"],
        ["--extra-args", "--hf-checkpoint /other"],
    ],
)
def test_recipe_rejects_overriding_owned_paths(node_args, argv):
    node_args.input_root, node_args.state_root = node_args.local_root / "inputs", node_args.local_root / "state"
    with pytest.raises(ValueError, match="owns"):
        recipe.make_launch(node_args, argv, resume_root=None)


def test_shell_quoted_launcher_arguments_are_preserved(node_args):
    parsed, argv = node_entry.parse_args(
        [
            "--volume-root",
            str(node_args.volume_root),
            "--run-id",
            "run",
            "--launcher-path",
            "example.launch",
            "--launcher-args=--extra-args '--lr 2e-6 --lr-warmup-iters 2' --save-hf",
        ]
    )
    assert argv == ["--extra-args", "--lr 2e-6 --lr-warmup-iters 2", "--save-hf"]
    assert parsed.num_nodes == 8


def test_bootstrap_argv_has_external_ray_allocation(node_args):
    head = node_entry.ray_argv(node_args, head=True)
    worker = node_entry.ray_argv(node_args, head=False)
    assert "--head" in head and "--num-gpus=8" in head
    assert "--address=127.0.0.1:6379" in worker and "--head" not in worker


def test_workers_return_the_heads_nonzero_outcome(node_args, tmp_path):
    root = tmp_path / "outcome"
    root.mkdir()
    (root / "result.json").write_text('{"exit_code":7}')
    assert node_entry.run_worker(node_args, root) == 7


def test_head_failure_writes_result_after_cleanup(node_args, monkeypatch):
    stopped = []

    def fail(*args):
        raise RuntimeError("launcher failed")

    monkeypatch.setattr(node_entry, "run_head", fail)
    monkeypatch.setattr(node_entry.ray, "is_initialized", lambda: False)
    monkeypatch.setattr(node_entry.subprocess, "run", lambda argv, **kwargs: stopped.append(argv))
    assert (
        node_entry.main(
            ["--volume-root", str(node_args.volume_root), "--run-id", "run", "--launcher-path", "example.launch"]
        )
        == 1
    )
    result = node_args.volume_root / "runs/run/attempts/attempt/result.json"
    assert stopped == [["ray", "stop", "--force"]]
    assert stage_models.read_json(result) == {"exit_code": 1, "error": "RuntimeError: launcher failed"}


def test_logs_publish_while_child_is_running_and_flush_at_exit(tmp_path):
    source, target = tmp_path / "launcher.log", tmp_path / "volume/logs"
    source.write_bytes(b"start\n")
    with process.LogPublisher(source, target):
        deadline = time.monotonic() + 2
        while not list(target.glob("*.log")) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert list(target.glob("*.log"))
        with source.open("ab") as stream:
            stream.write(b"end\n")
    assert b"".join(path.read_bytes() for path in sorted(target.glob("*.log"))) == b"start\nend\n"


def test_log_retry_keeps_the_original_chunk_when_log_grows(tmp_path, monkeypatch):
    source, target = tmp_path / "launcher.log", tmp_path / "volume/logs"
    source.write_bytes(b"first")
    log = process.LogPublisher(source, target)
    writes = []
    original = process.fuse.write_bytes

    def write(path, data, **kwargs):
        writes.append(data)
        if len(writes) == 1:
            raise OSError("interrupted log copy")
        return original(path, data, **kwargs)

    monkeypatch.setattr(process.fuse, "write_bytes", write)
    with pytest.raises(OSError):
        log.flush()
    with source.open("ab") as stream:
        stream.write(b"second")
    log.flush()
    assert writes[:2] == [b"first", b"first"]
    assert b"".join(path.read_bytes() for path in sorted(target.glob("*.log"))) == b"firstsecond"


def test_log_shutdown_preserves_bytes_written_during_a_copy(tmp_path, monkeypatch):
    source, target = tmp_path / "launcher.log", tmp_path / "volume/logs"
    source.write_bytes(b"first")
    copied, release = threading.Event(), threading.Event()
    original = process.fuse.write_bytes

    def write(path, data, **kwargs):
        original(path, data, **kwargs)
        if not copied.is_set():
            copied.set()
            assert release.wait(timeout=2)

    monkeypatch.setattr(process.fuse, "write_bytes", write)
    with process.LogPublisher(source, target) as log:
        assert copied.wait(timeout=2)
        with source.open("ab") as stream:
            stream.write(b"second")
        log._stop.set()
        release.set()
    assert b"".join(path.read_bytes() for path in sorted(target.glob("*.log"))) == b"firstsecond"
