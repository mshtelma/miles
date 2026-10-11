"""Run only a Miles launcher's ``prepare`` step, with the launcher's own flags.

    python -m tools.air.prepare scripts/run_qwen3_4b.py --model-dir /tmp/miles-air/models --data-dir /tmp/miles-air/data

Most launchers download and convert in ``prepare`` and run it again before every job. tools/air/prepare.yaml runs it
once on one node and copies its outputs to a Volume, from which every node of a training job stages them.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import typer

import miles.utils.external_utils.command_utils as U


def load_launcher(path: Path):
    """Import a launcher as running it would: its own directory first on ``sys.path``."""
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_command(module):
    """The launcher's ``prepare`` as a CLI command: as it is, or turned into one from its ``ScriptArgs``."""
    prepare = getattr(module, "prepare", None)
    if prepare is None:
        raise SystemExit(f"{module.__name__} has no prepare step")
    first = next(iter(inspect.signature(prepare).parameters.values()), None)
    return U.dataclass_cli(prepare) if first is not None and first.name == "args" else prepare


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m tools.air.prepare LAUNCHER [launcher flags]")
    launcher, *flags = argv
    app = typer.Typer(add_completion=False)
    app.command()(prepare_command(load_launcher(Path(launcher))))
    app(args=flags, prog_name=launcher)


if __name__ == "__main__":
    main()
