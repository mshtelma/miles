import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "job",
    [
        "examples/glm5_3_flash/officeqa/air/prepare.yaml",
        "examples/glm5_3_flash/officeqa/air/train.yaml",
        "examples/glm5_3_flash/officeqa/air/eval.yaml",
        "tools/air/train.yaml",
    ],
)
def test_job_is_omegaconf_and_shell_compatible(job):
    path = _ROOT / job
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert (path.parent / config["code_source"]["snapshot"]["root_path"]).resolve() == _ROOT
    subprocess.run(["bash", "-n"], input=config["command"], text=True, check=True)
    assert "${VAR:-$(" not in config["command"]
    assert "${ARR[@]}" not in config["command"]
