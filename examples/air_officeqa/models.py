"""The models the OfficeQA recipe trains: their checkpoints, and how one SGLang engine serves each.

``run_officeqa.py`` adds how to train each model, and ``eval/server.py`` serves a checkpoint the way
the rollout engines do. The chat template and the SGLang parsers follow from the renderer
(``miles.utils.chat_template_utils.resolve_fixed_chat_template`` and ``resolve_reasoning_and_tool_call_parser``).
"""

from dataclasses import dataclass
from typing import Literal

ModelName = Literal["GLM-5.3-Flash", "Qwen3.5-35B-A3B"]


@dataclass(frozen=True)
class Model:
    name: ModelName  # also the HF checkpoint's directory under the model directory
    hf_repo: str
    hf_revision: str
    megatron_model_type: str  # scripts/models/<type>.py
    tito_model: str  # the session server's renderer (--tito-model)
    engine_gpus: int  # GPUs per SGLang engine, its tensor- and expert-parallel size
    sglang_args: tuple[str, ...]  # the model's own engine flags, as Miles takes them

    @property
    def torch_dist(self) -> str:
        """The Megatron conversion's directory under the checkpoint directory."""
        return f"{self.megatron_model_type}_torch_dist"


MODELS: dict[str, Model] = {
    model.name: model
    for model in (
        Model(
            name="GLM-5.3-Flash",
            hf_repo="zai-org/GLM-5.3-Flash",
            hf_revision="eb9eb208eb0d988989d07a6a12d0fdeb5f52574a",
            megatron_model_type="glm5.3-flash",
            tito_model="glm53",
            engine_gpus=8,
            # Its sparse attention (DSA) runs on TileLang. Routing replay (R3) needs a MoE backend that reports
            # the experts it routed to.
            sglang_args=(
                "--sglang-dsa-prefill-backend",
                "tilelang",
                "--sglang-dsa-decode-backend",
                "tilelang",
                "--sglang-moe-runner-backend",
                "triton",
            ),
        ),
        Model(
            name="Qwen3.5-35B-A3B",
            hf_repo="Qwen/Qwen3.5-35B-A3B",
            hf_revision="59d61f3ce65a6d9863b86d2e96597125219dc754",
            megatron_model_type="qwen3.5-35B-A3B",
            tito_model="qwen35",
            engine_gpus=8,
            # FlashAttention 3, SGLang's default on Hopper, is not in the AIR image.
            sglang_args=("--sglang-attention-backend", "flashinfer", "--sglang-moe-runner-backend", "triton"),
        ),
    )
}
