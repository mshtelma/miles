import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docker"))

import build  # noqa: E402


def _command(monkeypatch, *args, **kwargs) -> list[str]:
    commands = []
    monkeypatch.setattr(build, "run", lambda cmd, dry_run: commands.append(cmd))
    build.build_and_push(*args, **kwargs)
    return commands[0]


def test_the_air_image_builds_into_the_callers_registry_with_the_host_ca(monkeypatch):
    command = _command(
        monkeypatch,
        "air-glm53",
        "custom",
        False,
        "docker/Dockerfile",
        custom_tag="glm53",
        image="example.io/me/miles-air",
    )

    assert command[command.index("-f") + 1] == str(build.REPO_ROOT / "docker/air/Dockerfile")
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert command[command.index("--secret") + 1] == "id=host_ca,src=/etc/ssl/certs/ca-certificates.crt"
    assert command[command.index("-t") + 1] == "example.io/me/miles-air:glm53"


def test_other_variants_keep_their_image_and_take_no_secrets(monkeypatch):
    command = _command(monkeypatch, "cu13-x86", "custom", False, "docker/Dockerfile", custom_tag="pr-1")

    assert command[command.index("-t") + 1] == "radixark/miles:pr-1"
    assert "--secret" not in command
