import os

from tests.ci.ci_register import register_cuda_ci
from tests.ci.metric_history import register_ci_gate
from tests.e2e.megatron.test_glm47_flash._common import CaseConfig, execute, prepare

# 8 GPUs keep Megatron DeepEP with TP x PP x CP (a 4-rank DeepEP group per PP stage), which
# no 4-GPU case can hold. The former SGLang failure was BF16 experts without the DeepGEMM
# runner (see _common.py).
register_cuda_ci(est_time=1100, suite="stage-c-8-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])

register_ci_gate(metric_key="train/grad_norm")
register_ci_gate(metric_key="train/ppo_kl")
register_ci_gate(metric_key="train/train_rollout_logprob_abs_diff")
register_ci_gate(metric_key="train/train_rollout_kl")
register_ci_gate(metric_key="rollout/raw_reward")

CASE = CaseConfig(
    use_deepep=True,
    num_gpus_per_node=8,
    cp_size=2,
    pp_size=2,
    tp_size=2,
    ep_size=4,
    # SGLang DeepEP forces EP to the engine TP.
    sglang_ep_size=4,
    # GLM-4.7-Flash has 20 attention heads; non-EP SGLang TP must divide it.
    rollout_num_gpus_per_engine=4,
)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare(CASE)
    execute(CASE, wandb_file=__file__)
