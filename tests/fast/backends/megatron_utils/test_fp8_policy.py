"""The GLM FP8 export policy must agree with the pinned checkpoint's weight scales."""

import gzip
import json
from pathlib import Path

import pytest

from miles.utils.fp8_policy import build_fp8_skip_policy

_FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "glm5_next_fp8_inventory.json.gz"


@pytest.fixture(scope="module")
def inventory():
    with gzip.open(_FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def test_dotted_matching_avoids_layer_and_projection_collisions():
    policy = build_fp8_skip_policy(
        {"modules_to_not_convert": ["model.layers.1.self_attn", "mlp.gate"]},
        require_exclusions=True,
    )
    assert policy.keeps("model.layers.1.self_attn.kv_b_proj.weight")
    assert not policy.keeps("model.layers.10.self_attn.kv_b_proj.weight")
    assert policy.keeps("model.layers.10.mlp.gate.weight")
    assert not policy.keeps("model.layers.10.mlp.gate_up_proj.weight")


def test_glm_kept_modules_and_quantized_neighbors(inventory):
    policy = build_fp8_skip_policy(inventory["quantization_config"], require_exclusions=True)
    assert policy.keeps("model.layers.3.self_attn.kv_b_proj.weight")
    assert policy.keeps("model.layers.3.self_attn.indexer.wq_b.weight")
    assert policy.keeps("model.layers.0.self_attn.o_proj.weight")
    assert not policy.keeps("model.layers.3.self_attn.q_b_proj.weight")
    assert not policy.keeps("model.layers.10.mlp.experts.0.gate_proj.weight")


def test_missing_exclusions_fail_closed():
    with pytest.raises(ValueError, match="requires checkpoint modules_to_not_convert"):
        build_fp8_skip_policy({"quant_method": "fp8"}, require_exclusions=True)
