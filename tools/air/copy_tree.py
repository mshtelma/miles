"""Copy a directory tree with parallel shutil copies, to stage inputs onto a node and publish outputs.

    python -m tools.air.copy_tree SOURCE TARGET [--last RELATIVE_PATH ...]
"""

import argparse
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def copy_tree(source: Path, target: Path, *, last: tuple[str, ...] = (), workers: int = 8) -> None:
    """Copy every file under ``source`` into ``target``; the files named in ``last`` go after the rest."""
    if not source.is_dir():
        raise FileNotFoundError(f"no directory to copy: {source}")
    files = sorted(path for path in source.rglob("*") if path.is_file())
    deferred = [path for path in files if path.relative_to(source).as_posix() in last]

    def copy(path: Path) -> None:
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(copy, [path for path in files if path not in deferred]))
    for path in deferred:
        copy(path)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--last", action="append", default=[], help="a relative path to copy after the rest")
    args = parser.parse_args(argv)
    copy_tree(args.source, args.target, last=tuple(args.last))


if __name__ == "__main__":
    main()
