"""Check the per-export mHC scale lifecycle used by GLM weight sync."""

import pytest
import torch

from miles.backends.megatron_utils.megatron_to_hf.glm5_next import Glm5NextExportState


def test_export_state_emits_ordered_triples_and_fails_closed() -> None:
    state = Glm5NextExportState()
    site = "self_attention_hyper_connection"

    assert state.add_hc_scale("3", site, "alpha_post", torch.tensor(2.0)) == []
    with pytest.raises(ValueError, match="Duplicate"):
        state.add_hc_scale("3", site, "alpha_post", torch.tensor(9.0))
    assert state.add_hc_scale("3", site, "alpha_res", torch.tensor(3.0)) == []
    ((name, scale),) = state.add_hc_scale("3", site, "alpha_pre", torch.tensor(1.0))
    assert name == "model.layers.3.hc_attn_scale"
    assert scale.tolist() == [1.0, 2.0, 3.0]

    state.add_hc_scale("4", site, "alpha_pre", torch.tensor(1.0))
    with pytest.raises(ValueError, match="Incomplete"):
        state.finish()
    state.abort()
    with pytest.raises(RuntimeError, match="closed"):
        state.add_hc_scale("5", site, "alpha_pre", torch.tensor(1.0))
