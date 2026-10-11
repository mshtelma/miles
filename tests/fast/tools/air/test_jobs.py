import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[4]
_JOBS = [
    "examples/air_officeqa/air/prepare.yaml",
    "examples/air_officeqa/air/train.yaml",
    "examples/air_officeqa/air/eval.yaml",
    "tools/air/train.yaml",
]


@pytest.mark.parametrize("job", _JOBS)
def test_job_is_omegaconf_and_shell_compatible(job):
    path = _ROOT / job
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert (path.parent / config["code_source"]["snapshot"]["root_path"]).resolve() == _ROOT
    subprocess.run(["bash", "-n"], input=config["command"], text=True, check=True)
    assert "${VAR:-$(" not in config["command"]
    assert "${ARR[@]}" not in config["command"]


@pytest.mark.parametrize("job", _JOBS)
def test_job_does_not_retry(job):
    # AIR retries three times unless told otherwise, and a failed attempt would fail the same way.
    assert OmegaConf.load(_ROOT / job).max_retries == 0
