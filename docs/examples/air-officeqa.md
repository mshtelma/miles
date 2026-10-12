---
title: "Agentic RL on OfficeQA on Databricks AI Runtime"
description: "Agentic RL of GLM-5.3-Flash or Qwen3.5-35B-A3B on OfficeQA at 150K tokens on Databricks AI Runtime, from the published model to an evaluated checkpoint."
# Generated from examples/air_officeqa/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
This example trains a research agent on [OfficeQA](https://huggingface.co/datasets/databricks/officeqa).
The agent answers questions about U.S. Treasury Bulletins: it searches and reads the bulletins with four tools,
computes in a Python sandbox, and submits an answer with the path of evidence behind it. An LLM judge must
confirm that the path supports the answer before the answer earns reward. Training is fully async DAPO at
150K tokens per episode, with either of two models:

| `MODEL_NAME` | GPUs | Trainer | Rollout engines |
| --- | --- | --- | --- |
| `GLM-5.3-Flash` (the default) | eight nodes of 8x B300 | four nodes: TP1, CP8, PP4, EP8 | four, one per node |
| `Qwen3.5-35B-A3B` | three nodes of 8x H100 | two nodes: TP2 with sequence parallelism, CP8, EP8 | one |

Qwen3.5-35B-A3B runs the same agent, tools and reward on 24 H100s. Before any training it solves 19% of the
held-out questions on the first try (GLM-5.3-Flash about 40%), so its reward has a signal from the first step.
[`models.py`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/models.py) holds each model's checkpoint and engine flags, and
[`run_officeqa.py`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/run_officeqa.py) its trainer layout.

This page runs the whole flow on Databricks AI Runtime, from the published model to an evaluated checkpoint,
and on your own Ray cluster. The [GLM-5.3-Flash page](/models/glm/glm5-3-flash) explains the recipe's
settings, and [Miles on Databricks AI Runtime](https://github.com/radixark/miles/blob/main/tools/air/README.md) how Miles runs on AIR in general,
other launchers and tasks included.

## How it runs

Three AIR jobs, in order:

| Job | Nodes | Time, GLM-5.3-Flash / Qwen3.5-35B-A3B | What it leaves in `VOLUME_ROOT` |
| --- | --- | --- | --- |
| [`prepare.yaml`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/air/prepare.yaml) | 1 | ~40 min / ~11 min | `data/officeqa/`: prompts and corpus; `models/<MODEL_NAME>/`: the HF checkpoint; `ckpt/<megatron model type>_torch_dist/`: its Megatron conversion |
| [`train.yaml`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/air/train.yaml) | 8 / 3 | ~45 min / ~40 min for the first step, then ~5 min / ~12 min a step | `runs/<RUN_ID>/checkpoints/`: one save every 10 steps, each with a drop-in HF checkpoint in `iter_<step>/hf` |
| [`eval.yaml`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/air/eval.yaml) | 1 | ~1 h / ~1.5 h | `runs/<EVAL_RUN_ID>/<mlflow-run-id>/`: rewards, pass@k and trajectories |

`VOLUME_ROOT` is a directory on a Unity Catalog Volume, and every job reads and writes only there. One
`VOLUME_ROOT` holds both models. All three jobs log to the MLflow experiment `air-officeqa`.

## Before you start

You need:

* **The `air` CLI** for Databricks AI Runtime, with a Databricks CLI profile for your workspace:
  `databricks auth login --profile <profile>`.
* **AI Runtime with the model's GPUs:** eight `GPU_8xB300` nodes for GLM-5.3-Flash, or three `GPU_8xH100` nodes
  for Qwen3.5-35B-A3B.
* **A Volume directory** you can write, for example `/Volumes/<catalog>/<schema>/<volume>/miles`.
* **Hugging Face access to the gated dataset.** Accept the terms of
  [`databricks/officeqa`](https://huggingface.co/datasets/databricks/officeqa), then store a token of that
  account in a secret. Any scope and key work, for example:

  ```bash
  databricks secrets create-scope <scope> -p <profile>
  databricks secrets put-secret <scope> <key> -p <profile>   # paste the token when prompted
  ```

* **The judge's serving endpoints**, `databricks-claude-opus-4-6` and `databricks-claude-sonnet-4-6`
  ([`judges.json`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/judges.json)). The jobs call them with your identity.
* **A container registry that AIR can pull from**, and Docker to build the image once. One image runs both
  models. The build pulls the Miles and AIR base images and takes about an hour. Register the digest the push
  prints:

  ```bash
  python docker/build.py --variant air-glm53 --image-tag custom --custom-tag glm53 \
    --image <registry>/miles-air --push
  air -p <profile> register image <registry>/miles-air@sha256:<digest>
  ```

  Where the build can't reach PyPI, add `--build-arg PIP_INDEX_URL=<mirror>`.

Every command below runs from the repository root and starts with these settings:

```bash
AIR=examples/air_officeqa/air
COMMON=(environment.docker_image.url=<registry>/miles-air@sha256:<digest>
  env_variables.VOLUME_ROOT=/Volumes/<catalog>/<schema>/<volume>/miles)
# GLM-5.3-Flash on B300, the jobs' default:
MODEL=()
TRAIN_GPUS=()
# or Qwen3.5-35B-A3B on H100:
MODEL=(env_variables.MODEL_NAME=Qwen3.5-35B-A3B compute.accelerator_type=GPU_8xH100)
TRAIN_GPUS=(compute.num_accelerators=24)
```

`air run` uploads your working tree, uncommitted changes included. To run a commit instead, add
`code_source.snapshot.git.commit=$(git rev-parse HEAD)` to the overrides. It prints the job's run ID, which
`air get run`, `air logs` and `air cancel` take.

## 1. Prepare the inputs

```bash
air -p <profile> run $AIR/prepare.yaml --override "${COMMON[@]}" "${MODEL[@]}" secrets.HF_TOKEN=<scope>/<key>
```

[`prepare_data.py`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/prepare_data.py) downloads OfficeQA at a pinned revision: the
questions, and the bulletins as text. It puts all 113 easy questions and 90 of the 133 hard ones in
`train.jsonl`, and holds out the other 43 hard ones in `eval_heldout.jsonl`. It also cuts the bulletins into
the passages the agent's search ranks. Then `examples/air_officeqa/run_officeqa.py prepare` downloads the
model's pinned revision and converts it to Megatron's `torch_dist` format.

## 2. Train

```bash
air -p <profile> run $AIR/train.yaml --override "${COMMON[@]}" "${MODEL[@]}" "${TRAIN_GPUS[@]}" \
  env_variables.RUN_ID=officeqa-run1
```

The defaults train 100 steps of 8 prompts x 8 samples and save every 10 steps, which takes about ten hours
for GLM-5.3-Flash and about twenty for Qwen3.5-35B-A3B. `TRAIN_ARGS` passes flags to
`examples/air_officeqa/run_officeqa.py train` (see its docstring) and replaces the default, `--save-hf`. For
a short check:

```bash
air -p <profile> run $AIR/train.yaml --override "${COMMON[@]}" "${MODEL[@]}" "${TRAIN_GPUS[@]}" \
  env_variables.RUN_ID=officeqa-check "env_variables.TRAIN_ARGS=--save-hf --num-rollout 2 --save-interval 2"
```

**Watch it.** In MLflow, the run named after `RUN_ID` has:
- the training metrics, such as `rollout/episode_raw_reward`, `curriculum/solve_rate`, `train/grad_norm`
  and `train/train_rollout_kl`;
- every Miles argument as a parameter;
- each node's GPU utilization;
- a `miles.save.<step>` tag with the path of each save.

The model page lists [what a healthy run looks like](/models/glm/glm5-3-flash#6-what-a-healthy-officeqa-run-looks-like).
The logs of every node are in the run's artifacts, and on the command line:

```bash
air -p <profile> logs <job-run-id> --tail 200           # the head node, which runs the trainer
air -p <profile> logs <job-run-id> --download-to logs/  # every node, complete
```

**Saves.** Each save is written to the nodes' local disks and copied to `runs/<RUN_ID>/checkpoints/` while
training goes on. The trainer waits for the save and its HF export: 10 to 12 minutes for GLM-5.3-Flash,
under 3 for Qwen3.5-35B-A3B. Only the newest save keeps its training state (4.4 TB for GLM-5.3-Flash,
486 GB for Qwen3.5-35B-A3B); the older ones keep just their HF checkpoint (328 GB and 72 GB).

**Resume.** The jobs don't retry on their own, because a failed attempt would most likely fail the same
way. Once the cause is fixed, submit the job again with the same `RUN_ID`, and it resumes from the newest
save. Keep `--num-rollout`, because Megatron rejects a different learning-rate schedule. Each submission is
its own MLflow run, named after `RUN_ID`. The job's 48-hour `timeout_minutes` includes the time spent
waiting for GPUs.

## 3. Evaluate

```bash
# the base model
air -p <profile> run $AIR/eval.yaml --override "${COMMON[@]}" "${MODEL[@]}" env_variables.EVAL_RUN_ID=base
# a save
air -p <profile> run $AIR/eval.yaml --override "${COMMON[@]}" "${MODEL[@]}" env_variables.EVAL_RUN_ID=run1-step100 \
  env_variables.HF_SOURCE=/Volumes/<catalog>/<schema>/<volume>/miles/runs/officeqa-run1/checkpoints/iter_0000099
```

The eval serves the checkpoint with a standalone SGLang server, set up like the rollout engines. It runs the
training agent, tools and reward on the 43 held-out questions, eight samples each, with as many episodes in
flight as one training engine serves. A save qualifies once its copy has finished and its HF export is
complete. `EVAL_ARGS` takes the flags of
[`eval/entry.py`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/eval/entry.py), such as `--samples-per-prompt 4` or `--limit-questions 3`.

Each eval writes these files to `runs/<EVAL_RUN_ID>/<mlflow-run-id>/`:
- `eval_results.json`: the rewards per question;
- `summary.json`: pass@k, questions solved and lost trajectories;
- `trajectories.jsonl`;
- the SGLang server's log.

Its MLflow run, named after `EVAL_RUN_ID`, has the same metrics. For a save, the held-out metrics, such as
`officeqa_eval/heldout/pass_at_1`, also go into the training run at the save's step, next to the training
curves.

When you no longer need a run, delete it: `databricks fs rm -r dbfs:<VOLUME_ROOT>/runs/<RUN_ID> -p <profile>`.

## Run it on your own cluster

On the model's nodes (eight 8-GPU nodes for GLM-5.3-Flash, three for Qwen3.5-35B-A3B) with the image's
software stack:

```bash
python examples/air_officeqa/run_officeqa.py prepare --model-name <model>   # one node
HF_TOKEN=<token> python -m examples.air_officeqa.prepare_data --out-dir /root/datasets/officeqa
# start Ray on all the nodes, then on the head node:
DATABRICKS_HOST=<workspace-url> DATABRICKS_TOKEN=<token> MILES_SCRIPT_EXTERNAL_RAY=1 \
  python examples/air_officeqa/run_officeqa.py train --model-name <model> --save-hf
```

The judge still calls the workspace's serving endpoints, so the workers need `DATABRICKS_HOST` and
`DATABRICKS_TOKEN`. Without `--persist-dir`, saves stay in the launcher's output directory, and nothing is
copied or resumed.

## What's in this directory

| Path | What it is |
| --- | --- |
| `run_officeqa.py` | The launcher: `prepare` and `train`, with each model's trainer layout |
| `models.py` | The two models: their checkpoints, and how one SGLang engine serves each |
| `agent.py`, `adapters.py` | The agent loop (model turns, tool calls, the final report) and its tool and reward adapters |
| `settings.py`, `tools.json` | The agent's limits and tools, shared by training and the eval |
| `reward.py` | The reward entry point, which scores the report |
| `prepare_data.py` | Downloads OfficeQA and builds the prompt files and the corpus |
| `corpus.py` | The bulletins, their search passages, and the four retrieval tools |
| `sandbox.py` | The Python sandbox, run with bubblewrap |
| `path_reward.py`, `path_report.py`, `judges.py`, `judges.json` | The answer and path checks, and the judge endpoints |
| `answer_check.py`, `answer_match.py` | The answer check, including OfficeQA's official scorer |
| `curriculum.py`, `eval_summary.py` | Samples questions by how often the policy solves them; pass rates and pass@k |
| `eval/` | The standalone SGLang eval that `eval.yaml` runs |
| `air/` | The three AIR jobs |
