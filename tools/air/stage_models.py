"""Inventories, publication and staging for AIR inputs, references and HF exports."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from tools.air import fuse_transfer as fuse

SCHEMA = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path: Path) -> dict:
    return json.loads(fuse.read_bounded(path))


def safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative or "\x00" in relative:
        raise ValueError(f"invalid artifact path: {relative!r}")
    if relative.startswith("/") or any(part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError(f"invalid artifact path: {relative!r}")
    return root / relative


def validate_files(files: list[dict]) -> dict[str, dict]:
    entries = {}
    for entry in files:
        name = entry["path"]
        safe_path(Path("/"), name)
        if name in entries or type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ValueError(f"duplicate path or invalid size: {name}")
        if not _SHA256.fullmatch(entry["sha256"]):
            raise ValueError(f"invalid SHA-256: {name}")
        entries[name] = entry
    return entries


def inventory(root: Path, paths=None) -> list[dict]:
    selected = list(paths) if paths is not None else sorted(root.rglob("*"))
    files = []
    for path in selected:
        if path.relative_to(root).parts[0] in {".cache", ".git"}:
            continue
        if path.is_symlink():
            raise ValueError(f"artifact contains a symlink: {path}")
        if not path.is_dir():
            files.append(path)

    def entry(path):
        size, digest = fuse.digest(path)
        return {"path": path.relative_to(root).as_posix(), "size_bytes": size, "sha256": digest}

    with ThreadPoolExecutor(max_workers=4) as pool:
        return sorted(pool.map(entry, files), key=lambda item: item["path"])


def copy_files(source: Path, target: Path, files: list[dict], limiter, *, upload: bool = False, repair=False) -> None:
    def copy(entry):
        with limiter.slot():
            src, dst = safe_path(source, entry["path"]), safe_path(target, entry["path"])
            expected = {key: entry[key] for key in ("size_bytes", "sha256")}
            if upload:
                fuse.upload_verified(src, dst, **expected, repair=repair)
            else:
                fuse.download_verified(src, dst, **expected)

    with ThreadPoolExecutor(max_workers=limiter.workers) as pool:
        list(pool.map(copy, files))


def write_complete(root: Path, document: dict) -> None:
    raw = json_bytes(document)
    marker = root / "COMPLETE.json"
    if fuse.exists(marker):
        try:
            previous = read_complete(root)
        except (ValueError, KeyError, FileNotFoundError):
            fuse.retry(marker.unlink, path=marker)
        else:
            if previous != document:
                raise ValueError(f"completed artifact differs from this publication: {root}")
            return
    fuse.write_bytes(root / "manifest.json", raw, repair=True)
    fuse.write_bytes(
        marker, json_bytes({"schema": SCHEMA, "manifest_sha256": hashlib.sha256(raw).hexdigest()}), repair=True
    )


def read_complete(root: Path) -> dict:
    complete = read_json(root / "COMPLETE.json")
    raw = fuse.read_bounded(root / "manifest.json")
    if complete.get("schema") != SCHEMA or complete.get("manifest_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError(f"COMPLETE does not match the manifest: {root}")
    document = json.loads(raw)
    if document.get("schema") != SCHEMA or not document.get("files"):
        raise ValueError(f"invalid artifact manifest: {root}")
    validate_files(document["files"])
    return document


def load_inputs(path: Path) -> dict:
    raw = fuse.read_bounded(path)
    if hashlib.sha256(raw).hexdigest() != path.stem:
        raise ValueError(f"input manifest differs from its SHA-256 filename: {path}")
    manifest = json.loads(raw)
    if manifest.get("schema") != SCHEMA or not manifest.get("artifacts"):
        raise ValueError(f"invalid input manifest: {path}")
    targets = set()
    for record in manifest["artifacts"]:
        _validate_artifact(record)
        validate_files(record["files"])
        if record["target"] in targets:
            raise ValueError(f"input manifest repeats a target: {record['target']}")
        targets.add(record["target"])
    return manifest


def _validate_artifact(record: dict) -> None:
    if record["kind"] not in {"model", "dataset"} or not _NAME.fullmatch(record["name"]):
        raise ValueError("artifact needs a model/dataset kind and a safe name")
    if not re.fullmatch(r"[0-9a-f]{40}", record["revision"]):
        raise ValueError("artifact revision must be an immutable 40-hex commit")
    expected = f"{'models' if record['kind'] == 'model' else 'data'}/{record['name']}"
    if record.get("target", expected) != expected:
        raise ValueError(f"artifact target differs from its identity: {record['name']}")


def _source(manifest: dict, record: dict) -> Path:
    source = record["source"]
    if source.startswith("/"):
        return Path(source)
    return safe_path(Path(manifest["volume_root"]), source)


def model_record(manifest: dict, name: str) -> dict:
    models = [r for r in manifest["artifacts"] if r["kind"] == "model" and r["name"] == name]
    if len(models) != 1:
        raise ValueError(f"input manifest must contain exactly one model named {name}")
    return models[0]


def prepare(spec_path: Path, datasets: dict[str, Path], volume_root: Path, local_root: Path) -> Path:
    records = []
    limiter = fuse.NodeTransferLimiter(local_root / "transfer-slots", 32)
    spec = read_json(spec_path)
    if spec["schema"] != SCHEMA:
        raise ValueError("invalid input spec schema")

    def publish(record, source, files=None):
        _validate_artifact(record)
        files = inventory(source) if files is None else files
        prefix = f"{'models' if record['kind'] == 'model' else 'datasets'}/{record['name']}@{record['revision']}"
        copy_files(source, volume_root / prefix, files, limiter, upload=True, repair=True)
        records.append(
            {
                **record,
                "source": prefix,
                "target": f"{'models' if record['kind'] == 'model' else 'data'}/{record['name']}",
                "files": files,
            }
        )

    local_root.mkdir(parents=True, exist_ok=True)
    for record in spec["artifacts"]:
        _validate_artifact(record)
        with tempfile.TemporaryDirectory(dir=local_root) as temporary:
            command = ["hf", "download", record["repo"], "--revision", record["revision"], "--local-dir", temporary]
            if record["kind"] == "dataset":
                command += ["--repo-type", "dataset"]
            subprocess.run(command, env={**os.environ, "HF_HUB_DISABLE_XET": "1"}, check=True)
            publish(record, Path(temporary))
    for name, source in datasets.items():
        files = inventory(source)
        revision = hashlib.sha256(json_bytes(files)).hexdigest()[:40]
        publish({"kind": "dataset", "name": name, "repo": f"local/{name}", "revision": revision}, source, files)
    manifest = {"schema": SCHEMA, "volume_root": str(volume_root), "artifacts": records}
    raw = json_bytes(manifest)
    path = volume_root / "inputs" / f"{hashlib.sha256(raw).hexdigest()}.json"
    fuse.write_bytes(path, raw)
    fuse.write_bytes(volume_root / "inputs/latest", f"{path}\n".encode(), overwrite=True)
    print(f"INPUT_MANIFEST={path}", flush=True)
    return path


def stage_inputs(manifest: dict, local_root: Path, model_name: str, limiter, *, weights=True, datasets=True) -> None:
    model_record(manifest, model_name)
    for record in manifest["artifacts"]:
        if record["kind"] == "model" and record["name"] != model_name or record["kind"] == "dataset" and not datasets:
            continue
        files = record["files"]
        if record["kind"] == "model" and not weights:
            files = [f for f in files if not f["path"].endswith((".safetensors", ".bin"))]
        copy_files(_source(manifest, record), safe_path(local_root, record["target"]), files, limiter)


def read_reference(root: Path, input_digest: str) -> dict:
    document = read_complete(root)
    if document.get("input_manifest_sha256") != input_digest:
        raise ValueError("reference was converted from a different input manifest")
    files = validate_files(document["files"])
    if not {"latest_checkpointed_iteration.txt", "release/.metadata", "release/metadata.json"} <= files.keys():
        raise ValueError("reference lacks its release tracker or metadata")
    return document


def publish_reference(source: Path, target: Path, input_digest: str, limiter) -> dict:
    # Checkpoint inspection needs torch; preparing HF inputs does not.
    from tools.air.dcp_stage import validate_checkpoint

    files = inventory(source)
    if (source / "latest_checkpointed_iteration.txt").read_bytes().strip() != b"release":
        raise ValueError("reference tracker must name release")
    validate_checkpoint(source / "release", files)
    copy_files(source, target, files, limiter, upload=True, repair=not fuse.exists(target / "COMPLETE.json"))
    document = {"schema": SCHEMA, "kind": "reference", "input_manifest_sha256": input_digest, "files": files}
    write_complete(target, document)
    fuse.write_bytes(target.parent / "latest", f"{target}\n".encode(), overwrite=True)
    return document


def stage_checkpoint(source: Path, target: Path, document: dict, limiter) -> None:
    files = [f for f in document["files"] if not f["path"].endswith(".distcp") and "/hf/" not in f["path"]]
    copy_files(source, target, files, limiter)
    # Common state may be embedded in a small DCP shard instead of common.pt.
    from tools.air.dcp_stage import MAX_COMMON_BYTES, common_files, read_metadata

    entries = validate_files(document["files"])
    metadata_files = [f for f in files if f["path"].endswith("/.metadata")]
    for entry in metadata_files:
        iteration = Path(entry["path"]).parent
        names = common_files(read_metadata(target / iteration))
        small = [
            entries[(iteration / name).as_posix()]
            for name in names
            if entries[(iteration / name).as_posix()]["size_bytes"] <= MAX_COMMON_BYTES
        ]
        copy_files(source, target, small, limiter)
    # These local controls also pin the inventory for every rank's read-plan cache.
    (target / "manifest.json").write_bytes(json_bytes(document))
    if fuse.exists(source / "COMPLETE.json"):
        (target / "COMPLETE.json").write_bytes(fuse.read_bounded(source / "COMPLETE.json"))


@dataclass(frozen=True)
class StagedModel:
    path: Path
    identity: dict


def stage_hf(
    source: str, manifest_path: Path, local_root: Path, model_name: str, limiter, *, datasets=True
) -> StagedModel:
    manifest = load_inputs(manifest_path)
    stage_inputs(manifest, local_root, model_name, limiter, weights=source == "base", datasets=datasets)
    if source == "base":
        return StagedModel(
            local_root / "models" / model_name, {"kind": "base", "input_manifest_sha256": manifest_path.stem}
        )
    if not source.startswith("checkpoint:/"):
        raise ValueError("HF_SOURCE must be base or checkpoint:<absolute save root>")
    root = Path(source.removeprefix("checkpoint:"))
    document = read_complete(root)
    step = document["rollout_id"]
    entries = validate_files(document["files"])
    receipt_name = f"iter_{step:07d}/hf_export.json"
    receipt = entries.get(receipt_name)
    if receipt is None:
        raise ValueError("save has no HF export receipt; enable --save-hf")
    fuse.verify(root / receipt_name, size_bytes=receipt["size_bytes"], sha256=receipt["sha256"])
    if read_json(root / receipt_name).get("status") != "ok":
        raise ValueError("save's HF export did not succeed")
    prefix = f"iter_{step:07d}/hf/"
    files = [{**f, "path": f["path"].removeprefix(prefix)} for f in document["files"] if f["path"].startswith(prefix)]
    if not files:
        raise ValueError("save has no HF export files")
    digest = read_json(root / "COMPLETE.json")["manifest_sha256"]
    target = local_root / "models" / f"hf-{digest[:16]}"
    copy_files(root / prefix, target, files, limiter)
    return StagedModel(
        target, {"kind": "checkpoint", "root": str(root), "rollout_id": step, "manifest_sha256": digest}
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepared = commands.add_parser("prepare")
    prepared.add_argument("--spec", type=Path, required=True)
    prepared.add_argument("--dataset", action="append", default=[])
    staged = commands.add_parser("stage")
    staged.add_argument("--manifest", type=Path, required=True)
    staged.add_argument("--model-name", required=True)
    published = commands.add_parser("publish-reference")
    published.add_argument("--source", type=Path, required=True)
    published.add_argument("--target", type=Path, required=True)
    published.add_argument("--input-manifest", type=Path, required=True)
    for command in (prepared, staged, published):
        command.add_argument("--local-root", type=Path, default=Path("/tmp/miles-air"))
    prepared.add_argument("--volume-root", type=Path, required=True)
    args = parser.parse_args(argv)
    limiter = fuse.NodeTransferLimiter(args.local_root / "transfer-slots", 32)
    if args.command == "prepare":
        datasets = {name: Path(path) for name, path in (value.split("=", 1) for value in args.dataset)}
        prepare(args.spec, datasets, args.volume_root, args.local_root)
    elif args.command == "stage":
        stage_inputs(load_inputs(args.manifest), args.local_root, args.model_name, limiter, datasets=False)
    else:
        publish_reference(args.source, args.target, args.input_manifest.stem, limiter)


if __name__ == "__main__":
    main()
