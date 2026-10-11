"""Every bridge of a model built on the delta-rule layer leaves the layout marker out of the HF weights."""

import ast
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_MBRIDGE = _ROOT / "miles_plugins/mbridge"
# The bridges of the models whose layers come from miles_plugins/models/linear_attn.py.
_DELTA_RULE_BRIDGES = ("glm5_next.py", "kimi_k3.py", "qwen3_5.py", "qwen3_next.py")


def _layout_module():
    spec = importlib.util.spec_from_file_location("linear_attn_layout_under_test", _MBRIDGE / "linear_attn_layout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_marker_is_left_out_and_every_weight_kept():
    class Base:
        def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep=True):
            self.consider_ep = consider_ep
            return {name: f"global.{name}" for name in model}

    class Bridge(_layout_module().SkipLinearAttnLayoutMarker, Base):
        pass

    marker = "decoder.layers.0.self_attention.linear_attn.weight_layout_version"
    weight = "decoder.layers.0.self_attention.linear_attn.in_proj_qkv.weight"
    bridge = Bridge()

    assert bridge._weight_name_mapping_mcore_local_to_global([marker, weight], consider_ep=False) == {
        weight: f"global.{weight}"
    }
    assert bridge.consider_ep is False


@pytest.mark.parametrize("bridge_file", _DELTA_RULE_BRIDGES)
def test_each_delta_rule_bridge_skips_the_marker(bridge_file):
    # Without it, converting the HF checkpoint fails: "Unsupported parameter name: ...weight_layout_version".
    tree = ast.parse((_MBRIDGE / bridge_file).read_text())
    registered = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(
            isinstance(d, ast.Call) and getattr(d.func, "id", None) == "register_model" for d in node.decorator_list
        )
    ]

    assert registered
    assert all(getattr(bridge.bases[0], "id", None) == "SkipLinearAttnLayoutMarker" for bridge in registered)


def test_the_bridge_list_covers_every_model_on_the_delta_rule_layer():
    models = _ROOT / "miles_plugins/models"
    users = {
        path.relative_to(models).parts[0].removesuffix(".py")
        for path in models.rglob("*.py")
        if path.name != "linear_attn.py" and "miles_plugins.models.linear_attn import" in path.read_text()
    }

    assert users == {name.removesuffix(".py") for name in _DELTA_RULE_BRIDGES}
