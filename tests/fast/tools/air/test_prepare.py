"""tools.air.prepare runs only a launcher's prepare step, with the launcher's own flags."""

import json
import sys

import pytest
from tools.air import prepare

_ARGS = """
import json
from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils as U


@dataclass
class ScriptArgs:
    model_dir: str = "/root/models"
    record: str = ""


def _record(args):
    with open(args.record, "w") as file:
        json.dump({"model_dir": args.model_dir}, file)
"""
# Most launchers: typer.run(main), where main prepares and then trains.
_MAIN_LAUNCHER = (
    _ARGS
    + """

def prepare(args: ScriptArgs):
    _record(args)


def execute(args: ScriptArgs):
    raise AssertionError("only prepare runs")


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
"""
)
# Launchers with one command per role, such as scripts/run_kimi_k3.py.
_APP_LAUNCHER = (
    _ARGS
    + """
app = typer.Typer()


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs):
    _record(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    raise AssertionError("only prepare runs")


@app.callback()
def _callback():
    pass
"""
)


@pytest.fixture(autouse=True)
def _restore_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.mark.parametrize("source", [_MAIN_LAUNCHER, _APP_LAUNCHER], ids=["main", "app"])
def test_only_prepare_runs_with_the_launchers_flags(tmp_path, source):
    launcher = tmp_path / "run_fake.py"
    launcher.write_text(source)
    record = tmp_path / "record.json"

    with pytest.raises(SystemExit) as done:
        prepare.main([str(launcher), "--model-dir", "/tmp/miles-air/models", "--record", str(record)])

    assert done.value.code in (0, None)
    assert json.loads(record.read_text()) == {"model_dir": "/tmp/miles-air/models"}


def test_a_launcher_without_a_prepare_step_is_refused(tmp_path):
    launcher = tmp_path / "run_fake.py"
    launcher.write_text(_ARGS)

    with pytest.raises(SystemExit, match="no prepare step"):
        prepare.main([str(launcher)])
