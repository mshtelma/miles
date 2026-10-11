# Miles on Databricks AI Runtime

Databricks AI Runtime (AIR) runs a command on serverless GPU nodes. This directory holds the two helpers Miles
uses there and a training job for any Miles launcher, and this page describes how Miles maps onto AIR. The
[GLM-5.3-Flash example](../../examples/glm5_3_flash/README.md) is a complete flow built from these pieces:
prepare, train and evaluate.

## How an AIR job runs

A job is a YAML file submitted with `air -p <profile> run job.yaml`; `--override KEY=VALUE ...` changes any
of its fields for one submission.

1. **The CLI uploads your code.** `code_source.snapshot` archives a local directory, by default the working
   tree, or a commit with `git.commit`. Every node unpacks it at `$CODE_SOURCE_PATH`.
2. **AIR starts the image on the nodes.** `compute.num_accelerators` counts GPUs: 64 `GPU_8xB300` GPUs are
   eight nodes. A custom image comes from a container registry: `air register image <image>@sha256:<digest>`
   registers it once, and `environment.docker_image.url` names it by that digest.
3. **Every node runs `command`,** with these variables set:
   - `NUM_NODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT` and `LOCAL_WORLD_SIZE`, for the rendezvous;
   - `MLFLOW_RUN_ID` and `MLFLOW_TRACKING_URI=databricks`, the attempt's MLflow run;
   - `DATABRICKS_HOST` and `DATABRICKS_TOKEN`, which act as you;
   - your `env_variables`, and your `secrets`, given as `NAME: <scope>/<key>`.
4. **The job ends when every node has exited.** It succeeds only if all exit 0. When one node fails, AIR
   stops the others and, up to `max_retries` times (three by default), starts a new attempt with a new
   MLflow run.

Each attempt's MLflow run is in the experiment named by `experiment_name`, under your workspace folder. It
holds each node's GPU and memory use as system metrics and each node's complete output as artifacts.
`air logs <job-run-id> --download-to DIR` fetches the logs, and `--retry N` picks an attempt.

## What Miles adds

**The image.** [`docker/air/Dockerfile`](../../docker/air/Dockerfile) moves the Miles stack (PyTorch,
Megatron-LM, SGLang, Ray) onto the `databricksruntime/air` base, which brings CUDA, NCCL and the network
drivers, and adds `databricks-sdk` and MLflow. Build and push it with
`python docker/build.py --variant air-glm53 --image-tag custom --custom-tag <tag> --image <registry>/miles-air --push`,
then register its digest. The variant also carries the GLM-5.3-Flash patches to SGLang and FLA, which other
models do not use.

**One Ray cluster across the nodes.** Every node joins it, and only the head node, `NODE_RANK` 0, runs the
launcher. `MILES_SCRIPT_EXTERNAL_RAY=1` makes a launcher submit to this cluster instead of starting its own.
The worker nodes block in `ray start` and exit 0 once the head stops Ray, so the head's exit code decides the
job.

**Inputs on local disk.** Volumes are mounted at `/Volumes/...` on every node. The engines and the agent's
tools read their inputs from local disk instead: [`copy_tree.py`](./copy_tree.py) copies a directory with
parallel file copies, for example the HF checkpoint and the data, to `/tmp` on each node before training:
`python -m tools.air.copy_tree SOURCE TARGET [--last RELATIVE_PATH ...]`. Files named with `--last` are
copied after the rest, so a reader that waits for them never sees a partial copy.

**Saves on the Volume, and resume.** Megatron saves to local disk. With
`--checkpoint-publisher-path tools.air.publisher.VolumePublisher` and `MILES_PERSIST_DIR` set to a Volume
directory, [`publisher.py`](./publisher.py) copies each save to that directory while training goes on, and
writes Megatron's tracker `latest_checkpointed_iteration.txt` last.
- **Resume:** `--load` pointed at the same directory resumes from the newest copied save. On a fresh start
  there is no tracker, and Miles loads `--ref-load` instead.
- **Failures:** the jobs set `max_retries: 0`, because a failed attempt would most likely fail the same
  way. Once the cause is fixed, resubmit the job; with the same `--load` it picks up where the last save
  left off.
- **Space:** only the newest save keeps its training state; older saves keep only their HF export.

**MLflow.** `--use-mlflow --mlflow-tracking-uri databricks --mlflow-run-id "$MLFLOW_RUN_ID" --mlflow-run-name <name>`
does three things:
- logs Miles' metrics and its arguments, as parameters, into the attempt's run;
- names the run;
- makes the publisher tag the run with each save's path (`miles.save.<step>`), and map each save to its run
  in `mlflow_runs.json` next to the tracker.

## Running a launcher on AIR

[`train.yaml`](./train.yaml) runs any launcher built on `ExecuteTrainConfig`, such as
`scripts/run_qwen3_4b.py`, with all of the above. Set its variables with `--override env_variables.NAME=...`:
- `LAUNCHER`: the script;
- `LAUNCHER_ARGS`: the launcher's own flags, such as `--mode debug_minimal`;
- `EXTRA_ARGS`: Miles flags, which come after the launcher's, such as `--num-rollout 4`;
- `INPUTS`: directories under `VOLUME_ROOT` that every node copies to `/tmp/miles-air` before training.

```bash
air -p <profile> run tools/air/train.yaml --override \
  environment.docker_image.url=<registry>/miles-air@sha256:<digest> \
  env_variables.VOLUME_ROOT=/Volumes/<catalog>/<schema>/<volume>/miles env_variables.RUN_ID=qwen3-4b-check \
  "env_variables.LAUNCHER_ARGS=--mode debug_minimal --model-dir /tmp/miles-air/models --data-dir /tmp/miles-air/data" \
  "env_variables.EXTRA_ARGS=--num-rollout 4 --sglang-attention-backend flashinfer"
```

The job passes the launcher two sets of flags:
- `--output-dir /tmp/miles-air/run`, which keeps the launcher's saves on local disk;
- `--extra-args` with `--load` set to `runs/<RUN_ID>/checkpoints`, the publisher, and the MLflow flags. Miles
  keeps the last `--load` it is given, so this one replaces the launcher's own.

The Qwen3 launchers download their model and data on the head node only, so they run on one node. For more
nodes, put the inputs on the Volume and list them in `INPUTS`. `scripts/run_glm5_3_flash.py` sets the same
flags through its `--persist-dir` and `--mlflow-run-id` options.

## Things to know

* **Time limits.** `timeout_minutes` covers all attempts together, including the time spent waiting for
  GPUs.
* **Idle GPUs.** AIR stops a job when any node's GPUs have all been idle for 60 minutes. Keep long loads,
  saves and copies under that.
* **Images.** They must be `linux/amd64` and under 20 GB. AIR ignores `WORKDIR`, so the command uses
  absolute paths or `cd "$CODE_SOURCE_PATH"`.
* **H100.** The image's kernels cover Hopper as well as Blackwell, except FlashAttention 3, which SGLang
  chooses by default on Hopper. On H100 nodes, add `--sglang-attention-backend flashinfer` to the Miles flags.
  To resume a run whose engines share the GPUs with the trainer (`--colocate`), also add
  `--sglang-mem-fraction-static 0.5`: the trainer loads its optimizer state while the engines hold their memory.
* **Identity.** Jobs act as the submitting user. `permissions` shares a run with others, and
  `usage_policy_name` bills it to a serverless usage policy.
* **The Databricks CLI.** Version 1.19 has the same commands as `databricks air`. It takes images from Unity
  Catalog instead (`environment.unity_catalog_image`, pushed with `databricks air images push`), and rejects
  `environment.docker_image`, so these jobs run with `air`. An AIR job is also a task type in Databricks
  Asset Bundles, `ai_runtime_task`.
