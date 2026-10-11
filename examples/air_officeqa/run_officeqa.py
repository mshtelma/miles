"""Train GLM-5.3-Flash or Qwen3.5-35B-A3B on OfficeQA with fully async agentic RL (DAPO, 150K-token context).

The OfficeQA agent (examples/air_officeqa) answers questions about U.S. Treasury Bulletins with four
retrieval tools and a Python sandbox, then submits a report; the reward checks the answer against the
gold one and asks an LLM judge whether the cited path supports it. The trainer takes the first nodes, and
every other node serves the policy as one 8-GPU SGLang engine:

  --model-name      sized for    trainer
  GLM-5.3-Flash     8x 8 B300    four nodes: TP1, CP8, PP4, EP8
  Qwen3.5-35B-A3B   3x 8 H100    two nodes: TP2 with sequence parallelism, CP8, EP8

What must exist before ``train``:
  * the HF checkpoint at ``{model_dir}/<model name>`` and its Megatron conversion at
    ``{ckpt_dir}/<megatron model type>_torch_dist`` (``prepare`` makes both);
  * the OfficeQA data at ``{data_dir}/officeqa``, written by
    ``python -m examples.air_officeqa.prepare_data``;
  * for more than one node, a joined Ray cluster (``MILES_SCRIPT_EXTERNAL_RAY=1``); on Databricks AI
    Runtime, examples/air_officeqa/air/train.yaml starts one;
  * for the OfficeQA reward, Databricks credentials for the judge endpoints in
    examples/air_officeqa/judges.json (``DATABRICKS_HOST`` and ``DATABRICKS_TOKEN`` are
    passed to the workers).

Args:
  --model-name: the model, one of examples/air_officeqa/models.py.
  --num-nodes: nodes in the Ray cluster; the trainer takes its nodes, the rest serve.
  --seq-length / --rollout-max-response-len: total context per episode, and tokens per model turn.
  --num-rollout / --rollout-batch-size / --n-samples-per-prompt: steps, prompts per step, samples each.
  --save-interval / --save-hf: full-state saves every N steps, each with a drop-in HF checkpoint
    (the trainer waits for the export; a GLM-5.3-Flash save takes 10 to 12 minutes with it).
  --persist-dir: save locally, copy every save there in the background (tools/air/publisher.py), and
    resume from the newest copied save; without one, start from the converted reference.
  --skip-download: ``prepare`` converts the checkpoint already at ``--hf-checkpoint``, without downloading.
  --mlflow-experiment / --mlflow-run-id: log to Databricks MLflow as well as wandb (if WANDB_API_KEY is
    set), into a new run of the experiment or into an existing run such as the AIR job's, which then
    takes the run id as its name.

Examples:
  python examples/air_officeqa/run_officeqa.py prepare
  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/air_officeqa/run_officeqa.py train --num-nodes 8 --save-hf
  MILES_SCRIPT_EXTERNAL_RAY=1 python examples/air_officeqa/run_officeqa.py train \
    --model-name Qwen3.5-35B-A3B --num-nodes 3
"""

import math
import os
import shlex
from dataclasses import dataclass, field

import typer
from examples.air_officeqa.models import MODELS, Model, ModelName
from examples.air_officeqa.settings import agent_env

import miles.utils.external_utils.command_utils as U
from miles.utils.chat_template_utils import resolve_reasoning_and_tool_call_parser

app = typer.Typer()


@dataclass(frozen=True)
class _Recipe:
    """How to train one model: the trainer's GPUs and layout, and its model-specific flags."""

    train_gpus: int
    tp: int
    cp: int
    pp: int
    ep: int
    default_nodes: int
    layout_args: tuple[str, ...]  # the rest of the trainer's layout
    convert_args: str  # the converter's layout; the trainer reshards the reference on load
    weight_check_skip: tuple[str, ...]  # tensors the initial weight-sync check leaves out
    model_args: tuple[str, ...]
    # Episodes in flight: what the engines' KV caches hold at once. More only queue, and a queued session loses
    # its cached prefix, so every later turn prefills its whole context again.
    max_concurrent_samples: int


_RECIPES: dict[str, _Recipe] = {
    # Four 8-GPU nodes train: TP1, CP8, PP4 (11/11/11/12 layers), EP8.
    "GLM-5.3-Flash": _Recipe(
        train_gpus=32,
        tp=1,
        cp=8,
        pp=4,
        ep=8,
        default_nodes=8,
        layout_args=(
            "--allgather-cp",
            "--decoder-first-pipeline-num-layers",
            "11",
            "--decoder-last-pipeline-num-layers",
            "12",
        ),
        convert_args=(
            "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 --expert-model-parallel-size 2 "
            "--expert-tensor-parallel-size 1 --decoder-first-pipeline-num-layers 11 "
            "--decoder-last-pipeline-num-layers 12"
        ),
        # The FP8 routed experts (the only FP8 tensors): the trainer loads the BF16 reference and re-quantizes on
        # export, so on the initial sync they differ from the engine's HF FP8 by about one ULP. BF16 tensors are
        # still checked exactly, which catches real mapping bugs.
        weight_check_skip=("visual.", "mlp.experts."),
        model_args=("--model-name", "glm5_next", "--qkv-format", "thd"),
        max_concurrent_samples=480,
    ),
    # Two 8-GPU nodes train: TP2 with sequence parallelism, CP8, EP8. A 150K-token episode is then about 19K tokens
    # a GPU; on one node, at 38K, the log-probability pass alone overflows an H100.
    "Qwen3.5-35B-A3B": _Recipe(
        train_gpus=16,
        tp=2,
        cp=8,
        pp=1,
        ep=8,
        default_nodes=3,
        # Log-probabilities over the 248K-token vocabulary are taken in chunks; whole, their logits do not fit.
        layout_args=("--sequence-parallel", "--log-probs-chunk-size", "4096"),
        convert_args="",
        weight_check_skip=("visual.",),
        # Megatron's own FlashAttention-2 path, as scripts/run_qwen3_5_35b_a3b_mtp.py uses off Blackwell. The model
        # definition adds the MTP block, whose loss Megatron takes over the whole vocabulary at once, even for
        # log-probabilities: at 150K tokens that alone overflows an H100. RL trains without it, and the HF export
        # carries the base model's MTP unchanged.
        model_args=("--attention-backend", "flash", "--mtp-num-layers", "0"),
        # One H100 engine holds about 68 OfficeQA sessions (2.4M tokens of KV cache): a step's 64 episodes.
        max_concurrent_samples=64,
    ),
}


def _max_tokens_per_gpu(seq_length: int, *, tp: int, cp: int) -> int:
    """Round allgather-CP capacity to a conservative local 1024-token quantum."""
    local_alignment = tp * 128
    quantum = math.lcm(local_alignment, 1024)
    global_alignment = cp * local_alignment
    padded = ((seq_length + global_alignment - 1) // global_alignment) * global_alignment
    return ((padded // cp + quantum - 1) // quantum) * quantum


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    model_name: ModelName = "GLM-5.3-Flash"
    # None: SLURM's allocation, else the recipe's node count.
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
    num_rollout: int = 100
    seq_length: int = 150000
    # Tokens per model turn; the episode as a whole is bounded by seq_length.
    rollout_max_response_len: int = 16384
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 8
    rollout_temperature: float = 1.0
    max_weight_staleness: int = 16
    # None: the recipe's.
    async_max_concurrent_samples: int | None = None
    async_max_prompt_attempts: int = 3
    async_retry_backoff_s: float = 2.0
    async_no_progress_timeout_s: float = 3000.0
    sglang_max_running_requests: int = 96
    sglang_mem_fraction_static: float = 0.7
    skip_saving: bool = False
    save_interval: int = 10
    save_hf: bool = False
    persist_dir: str | None = None
    mlflow_experiment: str | None = None
    mlflow_run_id: str | None = None
    extra_args: str = ""

    @property
    def model(self) -> Model:
        return MODELS[self.model_name]

    @property
    def recipe(self) -> _Recipe:
        return _RECIPES[self.model_name]

    @property
    def rollout_gpus(self) -> int:
        return self.num_nodes * self.num_gpus_per_node - self.recipe.train_gpus

    def __post_init__(self) -> None:
        if self.num_nodes is None:
            self.num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.recipe.default_nodes))
        if self.hf_checkpoint is None:
            self.hf_checkpoint = f"{self.model_dir}/{self.model_name}"
        if self.prompt_data is None:
            self.prompt_data = f"{self.data_dir}/officeqa/train.jsonl"
        if self.async_max_concurrent_samples is None:
            self.async_max_concurrent_samples = self.recipe.max_concurrent_samples
        self._validate()

    def _validate(self) -> None:
        if min(self.num_nodes, self.rollout_batch_size, self.num_rollout) <= 0:
            raise ValueError("node, batch and step counts must be positive")
        if self.num_gpus_per_node != 8:
            raise ValueError("the trainer layouts need whole eight-GPU nodes")
        if self.rollout_gpus <= 0 or self.rollout_gpus % self.model.engine_gpus:
            raise ValueError("the nodes beyond the trainer's must make whole rollout engines")
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
        if not 0 < self.sglang_mem_fraction_static < 1:
            raise ValueError("SGLang's static memory fraction must be in (0, 1)")


def _checkpoint_args(args: ScriptArgs) -> list[str]:
    flags = [
        "--hf-checkpoint",
        args.hf_checkpoint,
        "--ref-load",
        f"{args.ckpt_dir}/{args.model.torch_dist}",
        "--ckpt-format",
        "torch_dist",
    ]
    if args.persist_dir is not None:
        # Miles resumes when Megatron's tracker is there and starts from --ref-load otherwise.
        flags += ["--load", args.persist_dir, "--checkpoint-publisher-path", "tools.air.publisher.VolumePublisher"]
    if not _resuming(args):
        flags += ["--check-weight-update-equal", "--check-weight-update-skip-list", *args.recipe.weight_check_skip]
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
        "--metadata-key",
        "metadata",
        "--rollout-shuffle",
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
        "--rollout-top-p",
        "1.0",
        "--rollout-top-k",
        "-1",
        "--rollout-max-context-len",
        str(args.seq_length),
        "--rollout-max-prompt-len",
        str(args.seq_length - 1),
        "--rollout-max-response-len",
        str(args.rollout_max_response_len),
        "--data-source-path",
        "examples.air_officeqa.curriculum.CurriculumDataSource",
    ]


def _placement_args(args: ScriptArgs) -> list[str]:
    return [
        "--actor-num-nodes",
        str(args.recipe.train_gpus // args.num_gpus_per_node),
        "--actor-num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--num-gpus-per-node",
        str(args.num_gpus_per_node),
        "--rollout-num-gpus",
        str(args.rollout_gpus),
        "--pause-generation-mode",
        "retract",
        "--update-weight-transfer-mode",
        "broadcast",
        "--update-weights-interval",
        "1",
        "--fully-async",
        "--max-weight-staleness",
        str(args.max_weight_staleness),
        "--async-max-concurrent-samples",
        str(args.async_max_concurrent_samples),
        "--async-max-prompt-attempts",
        str(args.async_max_prompt_attempts),
        "--async-retry-backoff-s",
        str(args.async_retry_backoff_s),
        "--async-no-progress-timeout-s",
        str(args.async_no_progress_timeout_s),
    ]


def _parallel_args(args: ScriptArgs) -> list[str]:
    recipe = args.recipe
    return [
        "--seq-length",
        str(args.seq_length),
        "--tensor-model-parallel-size",
        str(recipe.tp),
        "--context-parallel-size",
        str(recipe.cp),
        "--pipeline-model-parallel-size",
        str(recipe.pp),
        "--expert-model-parallel-size",
        str(recipe.ep),
        "--expert-tensor-parallel-size",
        "1",
        "--max-tokens-per-gpu",
        str(_max_tokens_per_gpu(args.seq_length, tp=recipe.tp, cp=recipe.cp)),
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
        *recipe.layout_args,
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
        # Center group rewards without dividing by their spread (Dr. GRPO), so a group that differs
        # only by the judge's graded score does not get the push of a right-versus-wrong group.
        "--disable-grpo-std-normalization",
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
    model = args.model
    reasoning_parser, tool_call_parser = resolve_reasoning_and_tool_call_parser(model.tito_model)
    return [
        "--rollout-num-gpus-per-engine",
        str(model.engine_gpus),
        "--sglang-tp-size",
        str(model.engine_gpus),
        "--sglang-ep-size",
        str(model.engine_gpus),
        "--sglang-chunked-prefill-size",
        "8192",
        "--sglang-max-running-requests",
        str(args.sglang_max_running_requests),
        "--sglang-mem-fraction-static",
        str(args.sglang_mem_fraction_static),
        "--sglang-kv-cache-dtype",
        "bfloat16",
        *model.sglang_args,
        "--router-health-success-threshold",
        "1",
        "--router-health-check-interval-secs",
        "15",
        "--router-health-failure-threshold",
        "40",
        "--sglang-tool-call-parser",
        tool_call_parser,
        "--sglang-reasoning-parser",
        reasoning_parser,
    ]


def _agentic_args(args: ScriptArgs) -> list[str]:
    return [
        "--custom-generate-function-path",
        "miles.rollout.generate_hub.agentic_tool_call.generate",
        "--custom-agent-function-path",
        "examples.air_officeqa.agent.run",
        "--custom-rm-path",
        "examples.air_officeqa.reward.reward_func",
        "--use-session-server",
        # Each worker assembles its sessions' samples on one event loop; at 480 sessions in flight,
        # 32 workers left long episodes waiting past their collect deadline.
        "--session-server-workers",
        "64",
        "--tito-model",
        args.model.tito_model,
        "--session-addition-r3-on-retract",
        "--session-capture-provenance",
        "--max-seq-len",
        str(args.seq_length),
        "--agentic-abort-on-agent-error",
        "--async-unused-samples-handler",
        "retry",
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


def _misc_args(args: ScriptArgs) -> list[str]:
    return [
        "--attention-dropout",
        "0.0",
        "--hidden-dropout",
        "0.0",
        "--attention-softmax-in-fp32",
        "--accumulate-allreduce-grads-in-fp32",
        "--update-weight-buffer-size",
        str(1024**3),
        *args.recipe.model_args,
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
        **agent_env(args.data_dir),
        **({"MILES_PERSIST_DIR": args.persist_dir} if args.persist_dir is not None else {}),
    }
    for key in (
        "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
        "MLFLOW_RUN_ID",
        "MILES_ATTEMPT_ID",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
    ):
        if value := os.environ.get(key):
            values[key] = value
    return values


def _train_args(args: ScriptArgs) -> list[str]:
    blocks = (
        _checkpoint_args(args),
        _rollout_args(args),
        _placement_args(args),
        _parallel_args(args),
        _algorithm_args(),
        _optimizer_args(),
        _sglang_args(args),
        _agentic_args(args),
        _tracking_args(args),
        _misc_args(args),
        shlex.split(args.extra_args),
    )
    return [token for block in blocks for token in block]


def _train(args: ScriptArgs) -> None:
    U.execute_train(
        train_args=shlex.join(_train_args(args)),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.model.megatron_model_type,
        train_script="train_async.py",
        job_lifetime="launcher",
        extra_env_vars=_env_vars(args),
        megatron_path=args.megatron_path,
    )


def _prepare(args: ScriptArgs) -> None:
    model = args.model
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.ckpt_dir} {args.data_dir}")
    if not args.skip_download:
        U.exec_command_cpu(
            f"hf download {model.hf_repo} --revision {model.hf_revision} --local-dir {args.hf_checkpoint}"
        )
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    U.convert_checkpoint(
        model_name=model.megatron_model_type,
        megatron_model_type=model.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.ckpt_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
        extra_args=args.recipe.convert_args,
    )


@app.command()
@U.dataclass_cli
def prepare(args: ScriptArgs) -> None:
    """Download the HF checkpoint and convert it to torch_dist."""
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
