"""MLflow attachment to an explicitly selected existing run."""

import sys
import types
from argparse import Namespace

from miles.utils.tracking_utils import mlflow_utils


class _FakeMlflow(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("mlflow")
        self.experiments = []
        self.started = []
        self.runs = {
            "existing-run": Namespace(info=Namespace(experiment_id="existing-experiment", run_id="existing-run"))
        }

    def set_tracking_uri(self, uri) -> None:
        pass

    def set_experiment(self, experiment_name=None, experiment_id=None) -> None:
        self.experiments.append(experiment_id if experiment_id is not None else experiment_name)

    def get_run(self, run_id):
        return self.runs[run_id]

    def start_run(self, run_id=None, run_name=None, tags=None):
        self.started.append((run_id, tags))
        return self.runs.get(run_id) or Namespace(info=Namespace(run_id="new-run"))

    def log_params(self, params) -> None:
        pass


def _args(**overrides):
    values = {
        "use_mlflow": True,
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": "/Users/example/glm53",
        "mlflow_run_name": "run",
        "wandb_group": None,
        "rank": 0,
        "mlflow_run_id": None,
    }
    return Namespace(**{**values, **overrides})


def test_primary_attaches_in_the_explicit_runs_experiment(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(mlflow_utils, "_compute_config_for_logging", lambda args: {})
    monkeypatch.setenv("MLFLOW_RUN_ID", "existing-run")
    args = _args(mlflow_run_id="existing-run")
    mlflow_utils.init_mlflow(args, primary=True)
    assert fake.experiments == ["existing-experiment"]
    assert fake.started[0][0] == "existing-run"
    assert fake.started[0][1]["miles.requested_experiment"] == "/Users/example/glm53"
    # The existing run gets the requested name, which start_run alone would not set.
    assert fake.started[0][1]["mlflow.runName"] == "run"
    assert args.mlflow_run_id == "existing-run"

    mlflow_utils.init_mlflow(args, primary=False)
    assert fake.experiments[-1] == "existing-experiment" and fake.started[-1][0] == "existing-run"


def test_primary_without_explicit_run_uses_requested_experiment(monkeypatch):
    fake = _FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setattr(mlflow_utils, "_compute_config_for_logging", lambda args: {})
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    args = _args()
    mlflow_utils.init_mlflow(args, primary=True)
    assert fake.experiments == ["/Users/example/glm53"]
    assert fake.started[0][0] is None and "miles.requested_experiment" not in fake.started[0][1]
    assert "mlflow.runName" not in fake.started[0][1]
