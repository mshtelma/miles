# Miles on Databricks AI Runtime

Databricks AI Runtime (AIR) runs a command on serverless GPU nodes. This directory holds the two helpers Miles
uses there, and this page describes how any Miles launcher maps onto AIR. The
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
- **Retries:** a retry, or a resubmitted job with the same `--load`, picks up where the last save left off.
- **Space:** only the newest save keeps its training state; older saves keep only their HF export.

**MLflow.** `--use-mlflow --mlflow-tracking-uri databricks --mlflow-run-id "$MLFLOW_RUN_ID" --mlflow-run-name <name>`
does three things:
- logs Miles' metrics and its arguments, as parameters, into the attempt's run;
- names the run;
- makes the publisher tag the run with each save's path (`miles.save.<step>`), and map each save to its run
  in `mlflow_runs.json` next to the tracker.

## Running another launcher on AIR

[`examples/glm5_3_flash/officeqa/air/train.yaml`](../../examples/glm5_3_flash/officeqa/air/train.yaml) is
the template. A launcher built on `ExecuteTrainConfig`, such as `scripts/run_qwen3_30b_a3b.py`, takes the
same AIR wiring through its generic options:

```bash
LOCAL=/tmp/miles-air RUN="$VOLUME_ROOT/runs/$RUN_ID"
# ...copy the inputs to $LOCAL and start Ray as in train.yaml, then on the head node:
MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_qwen3_30b_a3b.py --num-nodes "$NUM_NODES" \
  --output-dir "$LOCAL/run" --model-dir "$LOCAL/models" --data-dir "$LOCAL/data" \
  --extra-env-vars "MILES_PERSIST_DIR=$RUN/checkpoints" \
  --extra-args "--load $RUN/checkpoints --checkpoint-publisher-path tools.air.publisher.VolumePublisher \
    --use-mlflow --mlflow-tracking-uri databricks --mlflow-run-id $MLFLOW_RUN_ID --mlflow-run-name $RUN_ID"
```

`--output-dir` keeps the launcher's saves on local disk, and the later `--load` in `--extra-args` replaces
the launcher's own. Only `scripts/run_glm5_3_flash.py` has run this way on AIR so far; its `--persist-dir`
and `--mlflow-run-id` options set the same flags.

## Things to know

* **Time limits.** `timeout_minutes` covers all attempts together, including the time spent waiting for
  GPUs.
* **Idle GPUs.** AIR stops a job when any node's GPUs have all been idle for 60 minutes. Keep long loads,
  saves and copies under that.
* **Images.** They must be `linux/amd64` and under 20 GB. AIR ignores `WORKDIR`, so the command uses
  absolute paths or `cd "$CODE_SOURCE_PATH"`.
* **Identity.** Jobs act as the submitting user. `permissions` shares a run with others, and
  `usage_policy_name` bills it to a serverless usage policy.
* **The Databricks CLI.** Version 1.19 has the same commands as `databricks air`. It takes images from Unity
  Catalog instead (`environment.unity_catalog_image`, pushed with `databricks air images push`), and rejects
  `environment.docker_image`, so these jobs run with `air`. An AIR job is also a task type in Databricks
  Asset Bundles, `ai_runtime_task`.
