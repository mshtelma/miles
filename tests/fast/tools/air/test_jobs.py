import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("job", ["prepare", "train", "eval"])
def test_job_is_omegaconf_and_shell_compatible(job):
    path = _ROOT / "examples/glm5_3_flash/officeqa/air" / f"{job}.yaml"
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    assert (path.parent / config["code_source"]["snapshot"]["root_path"]).resolve() == _ROOT
    subprocess.run(["bash", "-n"], input=config["command"], text=True, check=True)
    assert "${VAR:-$(" not in config["command"]
    assert "${ARR[@]}" not in config["command"]
