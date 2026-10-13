"""Train GLM-5.3-Flash with fully async DAPO on DAPO-Math-17k, the standard Miles math recipe.

On eight 8-GPU nodes, four nodes train (TP1, CP8, PP4 with 11/11/11/12 layers, EP8) and four serve the policy as
8-GPU SGLang engines, which keep generating while the trainer steps. The reward is the deepscaler answer check, and
AIME-2024 is evaluated every 20 steps. Another task is a matter of flags: ``--prompt-data`` for the prompts and
``--extra-args`` for the rest, such as ``--custom-rm-path``. examples/air_officeqa trains this layout on an agentic
task; tools/air runs this launcher on Databricks AI Runtime.

What must exist before ``train``:
  * the HF checkpoint at ``{model_dir}/GLM-5.3-Flash``, its Megatron conversion at
    ``{ckpt_dir}/glm5.3-flash_torch_dist``, and the datasets under ``{data_dir}`` (``prepare`` makes all three);
  * for more than one node, a joined Ray cluster (``MILES_SCRIPT_EXTERNAL_RAY=1``).

Args:
  --num-nodes: nodes in the Ray cluster; the trainer takes four, the rest serve, eight GPUs per engine.
  --seq-length / --rollout-max-response-len: tokens per sample, and per response.
  --num-rollout / --rollout-batch-size / --n-samples-per-prompt: steps, prompts per step, samples each.
  --prompt-data / --enable-eval: the training prompts, and the AIME-2024 eval every 20 steps.
  --save-interval / --save-hf: full-state saves every N steps, each with a drop-in HF checkpoint.
  --persist-dir: save locally, copy every save there in the background (tools/air/publisher.py), and resume from
    the newest copied save; without one, start from the converted reference.
  --skip-download: ``prepare`` converts the checkpoint already at ``--hf-checkpoint``, without downloading it.
  --mlflow-experiment / --mlflow-run-id: log to Databricks MLflow as well as wandb (if WANDB_API_KEY is set).

Examples:
  python scripts/run_glm5_3_flash.py prepare
  MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_glm5_3_flash.py train --num-nodes 8 --save-hf
"""

import math
import os
import shlex
from dataclasses import dataclass, field

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_HF_REPO = "zai-org/GLM-5.3-Flash"
_HF_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
_MODEL_TYPE = "glm5.3-flash"
# The trainer's layout: TP1, CP8, PP4 (11/11/11/12 layers), EP8 on four 8-GPU nodes.
_TP, _CP, _PP, _EP = 1, 8, 4, 8
_FIRST_STAGE_LAYERS, _LAST_STAGE_LAYERS = 11, 12
_TRAIN_GPUS = 32
_ENGINE_GPUS = 8
# The converter writes PP4 EP2; the trainer reshards the reference on load.
_CONVERT_ARGS = (
    "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 --expert-model-parallel-size 2 "
    "--expert-tensor-parallel-size 1 --decoder-first-pipeline-num-layers 11 --decoder-last-pipeline-num-layers 12"
)


def _max_tokens_per_gpu(seq_length: int) -> int:
    """Round allgather-CP capacity to a conservative local 1024-token quantum."""
    local_alignment = _TP * 128
    quantum = math.lcm(local_alignment, 1024)
    global_alignment = _CP * local_alignment
    padded = ((seq_length + global_alignment - 1) // global_alignment) * global_alignment
    return ((padded // _CP + quantum - 1) // quantum) * quantum


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    # None: SLURM's allocation, else eight nodes.
    num_nodes: int | None = None
    num_gpus_per_node: int = 8
    run_id: str = field(default_factory=U.create_run_id)
    model_dir: str = "/root/models"
    ckpt_dir: str = "/root/ckpt"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    hf_checkpoint: str | None = None
    # prepare: convert the HF checkpoint already at hf_checkpoint (for example one staged from a Volume).
    skip_download: bool = False
    prompt_data: str | None = None
    enable_eval: bool = True
    num_rollout: int = 3000
    seq_length: int = 32768
    rollout_max_response_len: int = 8192
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 8
    rollout_temperature: float = 1.0
    max_weight_staleness: int = 16
    async_max_concurrent_samples: int = 512
    sglang_max_running_requests: int = 128
    sglang_mem_fraction_static: float = 0.7
    skip_saving: bool = False
    save_interval: int = 20
    save_hf: bool = False
    persist_dir: str | None = None
    mlflow_experiment: str | None = None
    mlflow_run_id: str | None = None
    extra_args: str = ""

    @property
    def rollout_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node - _TRAIN_GPUS

    def __post_init__(self) -> None:
        if self.num_nodes is None:
            self.num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 8))
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/GLM-5.3-Flash"
        if self.prompt_data is None:
            self.prompt_data = f"{self.data_dir}/dapo-math-17k/dapo-math-17k.jsonl"
        self._validate()

    def _validate(self) -> None:
        if min(self.num_nodes, self.rollout_batch_size, self.num_rollout) <= 0:
            raise ValueError("node, batch and step counts must be positive")
        if self.num_gpus_per_node != 8:
            raise ValueError("the CP8 layout needs whole eight-GPU nodes")
        if self.rollout_gpus <= 0 or self.rollout_gpus % _ENGINE_GPUS:
            raise ValueError("the nodes beyond the trainer's must make whole eight-GPU rollout engines")
        if self.n_samples_per_prompt < 2:
            raise ValueError("group-relative RL needs at least two samples per prompt")
        if self.async_max_concurrent_samples < self.n_samples_per_prompt:
            raise ValueError("concurrency must admit a whole prompt group")
        if not 0 < self.rollout_max_response_len < self.seq_length:
            raise ValueError("the response cap must leave room for a prompt")
        if self.save_hf and self.skip_saving:
            raise ValueError("--save-hf writes its export inside a save")
        if self.save_interval <= 0 or self.max_weight_staleness < 0:
            raise ValueError("the save interval must be positive and the staleness cap nonnegative")


def _checkpoint_args(args: ScriptArgs) -> list[str]:
    flags = [
        "--hf-checkpoint",
        args.hf_checkpoint,
        "--ref-load",
        f"{args.ckpt_dir}/{_MODEL_TYPE}_torch_dist",
        "--ckpt-format",
        "torch_dist",
    ]
    if args.persist_dir is not None:
        # Miles resumes when Megatron's tracker is there and starts from --ref-load otherwise.
        flags += ["--load", args.persist_dir, "--checkpoint-publisher-path", "tools.air.publisher.VolumePublisher"]
    if not _resuming(args):
        # Skip the FP8 routed experts (the only FP8 tensors): the trainer loads the BF16 reference and
        # re-quantizes on export, so on the initial sync they differ from the engine's HF FP8 by about
        # one ULP. BF16 tensors are still checked exactly, which catches real mapping bugs.
        flags += ["--check-weight-update-equal", "--check-weight-update-skip-list", "visual.", "mlp.experts."]
    if not args.skip_saving:
        flags += ["--save", f"{args.output_dir}/checkpoints", "--save-interval", str(args.save_interval)]
    if args.save_hf:
        flags += [
            "--save-hf",
            f"{args.output_dir}/checkpoints/iter_{{rollout_id:07d}}/hf",
            "--save-hf-layout-path",
            "miles.backends.megatron_utils.hf_base_layout.base_layout",
        ]
    return flags


def _resuming(args: ScriptArgs) -> bool:
    return args.persist_dir is not None and os.path.exists(f"{args.persist_dir}/latest_checkpointed_iteration.txt")


def _rollout_args(args: ScriptArgs) -> list[str]:
    return [
        "--prompt-data",
        args.prompt_data,
        "--input-key",
        "prompt",
        "--label-key",
        "label",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--rm-type",
        "deepscaler",
        "--num-rollout",
        str(args.num_rollout),
        "--rollout-batch-size",
        str(args.rollout_batch_size),
        "--n-samples-per-prompt",
        str(args.n_samples_per_prompt),
        "--global-batch-size",
        str(args.rollout_batch_size * args.n_samples_per_prompt),
        "--num-steps-per-rollout",
        "1",
        "--rollout-temperature",
        str(args.rollout_temperature),
        "--rollout-max-response-len",
        str(args.rollout_max_response_len),
    ]


def _eval_args(args: ScriptArgs) -> list[str]:
    if not args.enable_eval:
        return []
    return [
        "--eval-interval",
        "20",
        "--eval-prompt-data",
        "aime24",
        f"{args.data_dir}/aime-2024/aime-2024.jsonl",
        "--n-samples-per-eval-prompt",
        "16",
        "--eval-max-response-len",
        "16384",
        "--eval-temperature",
        "0.6",
        "--eval-top-p",
        "0.95",
    ]


def _placement_args(args: ScriptArgs) -> list[str]:
    return [
        "--actor-num-nodes",
        str(_TRAIN_GPUS // args.num_gpus_per_node),
        "--actor-num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--rollout-num-gpus",
        str(args.rollout_gpus),
        # In-flight requests keep their routes through a weight update, which routing replay (R3) relies on.
        "--pause-generation-mode",
        "in_place",
        "--update-weight-transfer-mode",
        "broadcast",
        "--update-weights-interval",
        "1",
        "--fully-async",
        "--max-weight-staleness",
        str(args.max_weight_staleness),
        "--async-max-concurrent-samples",
        str(args.async_max_concurrent_samples),
    ]


def _parallel_args(args: ScriptArgs) -> list[str]:
    return [
        "--seq-length",
        str(args.seq_length),
        "--tensor-model-parallel-size",
        str(_TP),
        "--context-parallel-size",
        str(_CP),
        "--pipeline-model-parallel-size",
        str(_PP),
        "--expert-model-parallel-size",
        str(_EP),
        "--expert-tensor-parallel-size",
        "1",
        "--max-tokens-per-gpu",
        str(_max_tokens_per_gpu(args.seq_length)),
        "--use-dynamic-batch-size",
        "--recompute-granularity",
        "full",
        "--recompute-method",
        "uniform",
        "--recompute-num-layers",
        "1",
        "--micro-batch-size",
        "1",
        "--optimizer-cpu-offload",
        "--use-precision-aware-optimizer",
        "--optimizer-offload-fraction",
        "1.0",
        "--overlap-cpu-optimizer-d2h-h2d",
        "--allgather-cp",
        "--decoder-first-pipeline-num-layers",
        str(_FIRST_STAGE_LAYERS),
        "--decoder-last-pipeline-num-layers",
        str(_LAST_STAGE_LAYERS),
    ]


def _algorithm_args() -> list[str]:
    # DAPO: GRPO advantages, asymmetric clipping, token-level loss and zero-variance groups filtered out,
    # with truncated importance sampling against the rollout engine's log-probabilities.
    return [
        "--loss-type",
        "policy_loss",
        "--advantage-estimator",
        "grpo",
        "--kl-coef",
        "0",
        "--kl-loss-coef",
        "0",
        "--kl-loss-type",
        "low_var_kl",
        "--entropy-coef",
        "0",
        "--eps-clip",
        "0.2",
        "--eps-clip-high",
        "0.28",
        "--calculate-per-token-loss",
        "--use-tis",
        "--tis-clip-low",
        "0",
        "--tis-clip",
        "2.0",
        "--dynamic-sampling-filter-path",
        "miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter",
    ]


def _optimizer_args() -> list[str]:
    return [
        "--optimizer",
        "adam",
        "--lr",
        "1e-6",
        "--lr-decay-style",
        "constant",
        "--weight-decay",
        "0.1",
        "--adam-beta1",
        "0.9",
        "--adam-beta2",
        "0.98",
    ]


def _sglang_args(args: ScriptArgs) -> list[str]:
    return [
        "--rollout-num-gpus-per-engine",
        str(_ENGINE_GPUS),
        "--sglang-tp-size",
        str(_ENGINE_GPUS),
        "--sglang-ep-size",
        str(_ENGINE_GPUS),
        "--sglang-chunked-prefill-size",
        "8192",
        "--sglang-max-running-requests",
        str(args.sglang_max_running_requests),
        "--sglang-mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--sglang-kv-cache-dtype",
        "bfloat16",
        # GLM-5.3's sparse attention (DSA) runs on TileLang. Routing replay needs a MoE backend that reports
        # the experts it routed to.
        "--sglang-dsa-prefill-backend",
        "tilelang",
        "--sglang-dsa-decode-backend",
        "tilelang",
        "--sglang-moe-runner-backend",
        "triton",
        "--router-health-success-threshold",
        "1",
        "--router-health-check-interval-secs",
        "15",
        "--router-health-failure-threshold",
        "40",
    ]


def _tracking_args(args: ScriptArgs) -> list[str]:
    flags = shlex.split(U.get_default_wandb_args(__file__, run_id=args.run_id))
    if args.mlflow_experiment is None and args.mlflow_run_id is None:
        return flags
    flags += ["--use-mlflow", "--mlflow-tracking-uri", "databricks", "--mlflow-run-name", args.run_id]
    if args.mlflow_experiment is not None:
        flags += ["--mlflow-experiment-name", args.mlflow_experiment]
    if args.mlflow_run_id is not None:
        flags += ["--mlflow-run-id", args.mlflow_run_id]
    return flags


def _misc_args() -> list[str]:
    return [
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--attention-softmax-in-fp32",
        "--accumulate-allreduce-grads-in-fp32",
        "--update-weight-buffer-size",
        str(1024**3),
        "--model-name",
        "glm5_next",
        "--qkv-format",
        "thd",
        "--use-rollout-routing-replay",
        "--rollout-health-check-interval",
        "300",
        "--rollout-health-check-timeout",
        "300",
        "--distributed-timeout-minutes",
        "60",
    ]


def _env_vars(args: ScriptArgs) -> dict[str, str]:
    values = {
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "SGLANG_RETURN_ORIGINAL_LOGPROB": "0",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "MILES_RUN_ID": args.run_id,
        **({"MILES_PERSIST_DIR": args.persist_dir} if args.persist_dir is not None else {}),
    }
    for key in ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "MLFLOW_RUN_ID", "MILES_ATTEMPT_ID"):
        if value := os.environ.get(key):
            values[key] = value
    return values


def _train_args(args: ScriptArgs) -> list[str]:
    blocks = (
        _checkpoint_args(args),
        _rollout_args(args),
        _eval_args(args),
        _placement_args(args),
        _parallel_args(args),
        _algorithm_args(),
        _optimizer_args(),
        _sglang_args(args),
        _tracking_args(args),
        _misc_args(),
        shlex.split(args.extra_args),
    )
    return [token for block in blocks for token in block]


def _train(args: ScriptArgs) -> None:
    U.execute_train(
        train_args=shlex.join(_train_args(args)),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MODEL_TYPE,
        train_script="train_async.py",
        job_lifetime="launcher",
        extra_env_vars=_env_vars(args),
        megatron_path=args.megatron_path,
    )


def _prepare(args: ScriptArgs) -> None:
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.ckpt_dir} {args.data_dir}")
    if not args.skip_download:
        U.exec_command_cpu(f"hf download {_HF_REPO} --revision {_HF_REVISION} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    U.convert_checkpoint(
        model_name=_MODEL_TYPE,
        megatron_model_type=_MODEL_TYPE,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args=_CONVERT_ARGS,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the HF checkpoint and the datasets, and convert the checkpoint to torch_dist."""
    _prepare(args)


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs) -> None:
    """Launch the RL run."""
    _train(args)


# Without a callback, Typer collapses a two-command app into a single command.
@app.callback()
def _callback() -> None:
    pass


if __name__ == "__main__":
    app()
