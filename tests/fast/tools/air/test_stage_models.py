import pytest
from tests.fast.tools.air.conftest import copy_save

from tools.air import fuse_transfer as fuse
from tools.air import publisher, stage_models


def test_every_node_gets_full_hf_weights_and_datasets(input_manifest, tmp_path):
    for rank in (0, 1):
        local = tmp_path / f"node-{rank}"
        stage_models.stage_inputs(
            input_manifest[1], local, "GLM-5.3-Flash", fuse.NodeTransferLimiter(local / "slots", 2)
        )
        assert (local / "models/GLM-5.3-Flash/model.safetensors").read_bytes() == b"HF weights"
        assert (local / "data/officeqa/train.jsonl").is_file()
        assert (local / "data/officeqa/curriculum.json").is_file()


def test_staging_checks_actual_bytes_against_the_input_manifest(input_manifest, tmp_path):
    (tmp_path / "volume/models/model/model.safetensors").write_bytes(b"different bytes")
    with pytest.raises(ValueError, match="size/SHA-256 mismatch"):
        stage_models.stage_inputs(
            input_manifest[1], tmp_path / "local", "GLM-5.3-Flash", fuse.NodeTransferLimiter(tmp_path / "slots", 2)
        )
    assert not (tmp_path / "local/models/GLM-5.3-Flash/model.safetensors").exists()


def test_input_manifest_is_pinned_by_its_filename(input_manifest):
    path, manifest = input_manifest
    assert stage_models.load_inputs(path) == manifest
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256 filename"):
        stage_models.load_inputs(path)


def test_prepare_publishes_local_dataset_inventory_and_latest_pointer(tmp_path):
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "data.jsonl").write_text('{"prompt":"q"}\n')
    spec = tmp_path / "spec.json"
    spec.write_text('{"schema":1,"artifacts":[]}')
    path = stage_models.prepare(spec, {"officeqa": source}, tmp_path / "volume", tmp_path / "scratch")
    manifest = stage_models.load_inputs(path)
    record = manifest["artifacts"][0]
    assert (tmp_path / "volume/inputs/latest").read_text().strip() == str(path)
    assert record["files"] == stage_models.inventory(source)
    assert (tmp_path / "volume" / record["source"] / "data.jsonl").read_bytes() == (source / "data.jsonl").read_bytes()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "a//b", "a\\b", "a/./b"])
def test_manifest_paths_stay_inside_the_artifact(tmp_path, name):
    with pytest.raises(ValueError, match="invalid artifact path"):
        stage_models.safe_path(tmp_path, name)


def test_reference_checks_its_input_manifest(reference_root, input_manifest):
    root, document = reference_root
    assert stage_models.read_reference(root, input_manifest[0].stem) == document
    with pytest.raises(ValueError, match="different input manifest"):
        stage_models.read_reference(root, "0" * 64)


def test_fresh_reference_stages_controls_without_cursor_or_weight_shards(reference_root, tmp_path):
    root, document = reference_root
    local = tmp_path / "local/reference"
    stage_models.stage_checkpoint(root, local, document, fuse.NodeTransferLimiter(tmp_path / "slots", 2))
    assert (local / "latest_checkpointed_iteration.txt").read_text() == "release"
    assert (local / "release/.metadata").is_file()
    assert not list(local.rglob("*.distcp"))
    assert not (local / "rollout").exists()


def test_resume_controls_include_cursor_and_bounded_common_state(distributed_save, tmp_path):
    checkpoint = distributed_save()
    document = publisher.finalize_save(
        checkpoint.target, copy_save(checkpoint), run_id="run", attempt_id="attempt", rollout_id=3, configuration={}
    )
    local = tmp_path / "local/resume"
    stage_models.stage_checkpoint(checkpoint.target, local, document, fuse.NodeTransferLimiter(tmp_path / "slots", 2))
    assert (local / "rollout/global_dataset_state_dict_3.pt").is_file()
    assert (local / "iter_0000003/.metadata").is_file()
    assert (local / "COMPLETE.json").is_file()
    assert 0 < len(list(local.rglob("*.distcp"))) < len(list(checkpoint.target.rglob("*.distcp")))


@pytest.mark.parametrize("source", ["base", "checkpoint"])
def test_hf_staging_shares_inputs_and_requires_a_successful_export(source, input_manifest, distributed_save, tmp_path):
    local = tmp_path / "eval"
    limiter = fuse.NodeTransferLimiter(tmp_path / "slots", 2)
    if source == "checkpoint":
        checkpoint = distributed_save()
        iteration = checkpoint.roots["trainer"] / "iter_0000003"
        (iteration / "hf").mkdir()
        (iteration / "hf/opaque.bin").write_bytes(b"HF export")
        (iteration / "hf_export.json").write_text('{"status":"ok","future":{"field":1}}')
        publisher.finalize_save(
            checkpoint.target,
            copy_save(checkpoint),
            run_id="run",
            attempt_id="attempt",
            rollout_id=3,
            configuration={},
        )
        source = "checkpoint:" + str(checkpoint.target)
    staged = stage_models.stage_hf(source, input_manifest[0], local, "GLM-5.3-Flash", limiter)
    assert staged.path.is_dir()
    assert (local / "data/officeqa/train.jsonl").is_file()
    expected = b"HF weights" if source == "base" else b"HF export"
    assert (staged.path / ("model.safetensors" if source == "base" else "opaque.bin")).read_bytes() == expected


def test_failed_hf_receipt_is_ineligible_for_eval(input_manifest, distributed_save, tmp_path):
    checkpoint = distributed_save()
    (checkpoint.roots["trainer"] / "iter_0000003/hf_export.json").write_text('{"status":"failed"}')
    publisher.finalize_save(
        checkpoint.target, copy_save(checkpoint), run_id="run", attempt_id="attempt", rollout_id=3, configuration={}
    )
    with pytest.raises(ValueError, match="did not succeed"):
        stage_models.stage_hf(
            "checkpoint:" + str(checkpoint.target),
            input_manifest[0],
            tmp_path / "eval",
            "GLM-5.3-Flash",
            fuse.NodeTransferLimiter(tmp_path / "slots", 2),
        )
