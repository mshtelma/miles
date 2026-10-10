"""CPU regression tests for MTP detachment on bridge-built providers."""

import argparse
import ast
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])


@pytest.fixture(autouse=True)
def stub_dsa_backend_args(monkeypatch: pytest.MonkeyPatch) -> None:
    # The Bridge helper imports the DSA mapper; its Megatron import requires
    # libcuda in the AIR image, while these tests only check argument copying.
    module = ModuleType("miles.utils.megatron_bridge_utils")

    def ignore_backend_args(_provider, _args) -> None:
        return None

    module.apply_dsa_backend_args = ignore_backend_args
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture(scope="module")
def apply_bridge_runtime_config() -> Callable:
    # Execute the production helper without importing GPU-only Megatron modules.
    path = Path(__file__).resolve().parents[4] / "miles/backends/megatron_utils/model_provider.py"
    tree = ast.parse(path.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_apply_bridge_runtime_config"
    )
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_apply_bridge_runtime_config"]


@pytest.fixture
def runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=1,
        calculate_per_token_loss=True,
        variable_seq_lengths=True,
        attention_softmax_in_fp32=False,
        gradient_accumulation_fusion=False,
        fp32_residual_connection=False,
        deterministic_mode=False,
        recompute_granularity=None,
        recompute_method=None,
        recompute_num_layers=None,
        recompute_modules=[],
        cpu_offloading_num_layers=0,
        distribute_saved_activations=False,
        tp_comm_overlap=False,
        fp8=None,
        fp8_recipe=None,
        attention_backend="auto",
        moe_token_dispatcher_type="alltoall",
    )


@pytest.mark.parametrize(
    ("enabled", "initial_detach", "expected_detach"),
    [
        (True, False, True),
        (True, True, True),
        (False, False, False),
        (False, True, True),
        (None, False, False),
        (None, True, True),
    ],
)
def test_bridge_mtp_detachment(
    apply_bridge_runtime_config: Callable,
    runtime_args: argparse.Namespace,
    enabled: bool | None,
    initial_detach: bool,
    expected_detach: bool,
) -> None:
    # A missing flag covers callers that only register Megatron's arguments.
    if enabled is not None:
        runtime_args.enable_mtp_training = enabled
    provider = SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=initial_detach)

    apply_bridge_runtime_config(provider, runtime_args)

    assert provider.mtp_detach_heads is expected_detach
    assert provider.mtp_num_layers == 1


@pytest.mark.parametrize("provider_default", [False, True])
def test_bridge_forwards_an_explicit_no_rope_fusion(
    apply_bridge_runtime_config: Callable, runtime_args: argparse.Namespace, provider_default: bool
) -> None:
    runtime_args.apply_rope_fusion = False  # --no-rope-fusion
    provider = SimpleNamespace(apply_rope_fusion=provider_default)
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.apply_rope_fusion is False


@pytest.mark.parametrize("provider_default", [False, True])
def test_bridge_keeps_the_provider_choice_at_megatron_default(
    apply_bridge_runtime_config: Callable, runtime_args: argparse.Namespace, provider_default: bool
) -> None:
    """Megatron's parsed default is True; DeepSeek/Kimi/DSA providers set False and must not be overridden."""
    runtime_args.apply_rope_fusion = True
    provider = SimpleNamespace(apply_rope_fusion=provider_default)
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.apply_rope_fusion is provider_default


def test_bridge_preserves_rope_fusion_when_argument_is_absent(
    apply_bridge_runtime_config: Callable, runtime_args: argparse.Namespace
) -> None:
    provider = SimpleNamespace(apply_rope_fusion=True)
    apply_bridge_runtime_config(provider, runtime_args)
    assert provider.apply_rope_fusion is True
