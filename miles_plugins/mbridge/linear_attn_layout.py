"""The delta-rule layer's layout marker: checkpoint state, but no HF weight."""

# A persistent buffer of the head-sharded delta-rule layer (miles_plugins/models/linear_attn.py), saved in every
# DCP checkpoint so that a load can tell which layout the checkpoint was written in.
LAYOUT_MARKER_SUFFIX = ".self_attention.linear_attn.weight_layout_version"


class SkipLinearAttnLayoutMarker:
    """Leave the layout marker out of a bridge's weight names, for every model built on the delta-rule layer.

    mbridge maps each entry of the model's state dict, buffers included, to HF tensors; the marker has none.
    """

    def _weight_name_mapping_mcore_local_to_global(self, model, consider_ep: bool = True) -> dict[str, str]:
        mapping = super()._weight_name_mapping_mcore_local_to_global(model, consider_ep=consider_ep)
        return {local: name for local, name in mapping.items() if not local.endswith(LAYOUT_MARKER_SUFFIX)}
