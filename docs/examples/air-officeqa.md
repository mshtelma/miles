---
title: "GLM-5.3-Flash on OfficeQA"
description: "Agentic RL of GLM-5.3-Flash on OfficeQA at 150K tokens on Databricks AI Runtime, from the published model to an evaluated checkpoint in three jobs."
# Generated from examples/air_officeqa/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
This example trains GLM-5.3-Flash as a research agent on [OfficeQA](https://huggingface.co/datasets/databricks/officeqa).
The agent answers questions about U.S. Treasury Bulletins: it searches and reads the bulletins with four tools,
computes in a Python sandbox, and submits an answer with the path of evidence behind it. An LLM judge must
confirm that the path supports the answer before the answer earns reward. Training is fully async DAPO at
150K tokens per episode, on eight 8-GPU nodes.

This page runs the whole flow on Databricks AI Runtime, from the published model to an evaluated checkpoint,
and on your own Ray cluster. The [model page](/models/glm/glm5-3-flash) explains the model and the recipe's
settings, and [Miles on Databricks AI Runtime](https://github.com/radixark/miles/blob/main/tools/air/README.md) how Miles runs on AIR in general.

## How it runs

Three AIR jobs, in order:

| Job | Nodes (8x B300) | Time | What it leaves in `VOLUME_ROOT` |
| --- | --- | --- | --- |
| [`prepare.yaml`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/air/prepare.yaml) | 1 | ~40 min | `data/officeqa/`: prompts and corpus; `models/GLM-5.3-Flash/`: the HF checkpoint; `ckpt/glm5.3-flash_torch_dist/`: its Megatron conversion |
| [`train.yaml`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/air/train.yaml) | 8 | ~45 min for the first step, then ~5 min a step | `runs/<RUN_ID>/checkpoints/`: one save every 10 steps, each with a drop-in HF checkpoint in `iter_<step>/hf` |
| [`eval.yaml`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/air/eval.yaml) | 1 | ~1 h | `runs/<EVAL_RUN_ID>/<mlflow-run-id>/`: rewards, pass@k and trajectories |

`VOLUME_ROOT` is a directory on a Unity Catalog Volume, and every job reads and writes only there. All
three jobs log to the MLflow experiment `air-officeqa`.

## Before you start

You need:

* **The `air` CLI** for Databricks AI Runtime, with a Databricks CLI profile for your workspace:
  `databricks auth login --profile <profile>`.
* **AI Runtime with eight `GPU_8xB300` nodes.**
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
* **A container registry that AIR can pull from**, and Docker to build the image once. The build pulls the
  Miles and AIR base images and takes about an hour. Register the digest the push prints:

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
```

`air run` uploads your working tree, uncommitted changes included. To run a commit instead, add
`code_source.snapshot.git.commit=$(git rev-parse HEAD)` to the overrides. It prints the job's run ID, which
`air get run`, `air logs` and `air cancel` take.

## 1. Prepare the inputs

```bash
air -p <profile> run $AIR/prepare.yaml --override "${COMMON[@]}" secrets.HF_TOKEN=<scope>/<key>
```

[`prepare_data.py`](https://github.com/radixark/miles/blob/main/examples/air_officeqa/prepare_data.py) downloads OfficeQA at a pinned revision: the
questions, and the bulletins as text. It puts all 113 easy questions and 90 of the 133 hard ones in
`train.jsonl`, and holds out the other 43 hard ones in `eval_heldout.jsonl`. It also cuts the bulletins into
the passages the agent's search ranks. Then `examples/air_officeqa/run_officeqa.py prepare` downloads the pinned
GLM-5.3-Flash revision and converts it to Megatron's `torch_dist` format.

## 2. Train

```bash
air -p <profile> run $AIR/train.yaml --override "${COMMON[@]}" env_variables.RUN_ID=officeqa-run1
```

The defaults train 100 steps of 8 prompts x 8 samples and save every 10 steps, which takes about ten hours.
`TRAIN_ARGS` passes flags to `examples/air_officeqa/run_officeqa.py train` (see its docstring) and replaces the default,
`--save-hf`. For a short check:

```bash
air -p <profile> run $AIR/train.yaml --override "${COMMON[@]}" env_variables.RUN_ID=officeqa-check \
  "env_variables.TRAIN_ARGS=--save-hf --num-rollout 2 --save-interval 2"
```

**Watch it.** In MLflow, the run named after `RUN_ID` has:
- the training metrics, such as `rollout/episode_raw_reward`, `curriculum/solve_rate`, `train/grad_norm`
  and `train/train_rollout_kl`;
- every Miles argument as a parameter;
- each node's GPU utilization;
- a `miles.save.<step>` tag with the path of each save.

The model page lists [what a healthy run looks like](/models/glm/glm5-3-flash#6-what-a-healthy-run-looks-like).
The logs of every node are in the run's artifacts, and on the command line:

```bash
air -p <profile> logs <job-run-id> --tail 200           # the head node, which runs the trainer
air -p <profile> logs <job-run-id> --download-to logs/  # every node, complete
```

**Saves.** Each save is written to the nodes' local disks and copied to `runs/<RUN_ID>/checkpoints/` while
training goes on. A save's HF export holds the trainer 10 to 12 minutes. Only the newest save keeps its 4.4 TB
training state; the older ones keep just their 328 GB HF checkpoint.

**Resume.** The jobs don't retry on their own, because a failed attempt would most likely fail the same
way. Once the cause is fixed, submit the job again with the same `RUN_ID`, and it resumes from the newest
save. Keep `--num-rollout`, because Megatron rejects a different learning-rate schedule. Each submission is
its own MLflow run, named after `RUN_ID`. The job's 48-hour `timeout_minutes` includes the time spent
waiting for GPUs.

## 3. Evaluate

```bash
# the base model
air -p <profile> run $AIR/eval.yaml --override "${COMMON[@]}" env_variables.EVAL_RUN_ID=base
# a save
air -p <profile> run $AIR/eval.yaml --override "${COMMON[@]}" env_variables.EVAL_RUN_ID=run1-step100 \
  env_variables.HF_SOURCE=/Volumes/<catalog>/<schema>/<volume>/miles/runs/officeqa-run1/checkpoints/iter_0000099
```

The eval serves the checkpoint with a standalone SGLang server, set up like the rollout engines. It runs the
training agent, tools and reward on the 43 held-out questions, eight samples each. A save qualifies once its
copy has finished and its HF export is complete. `EVAL_ARGS` takes the flags of
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

On eight 8-GPU nodes with the image's software stack:

```bash
python examples/air_officeqa/run_officeqa.py prepare                                    # one node
HF_TOKEN=<token> python -m examples.air_officeqa.prepare_data --out-dir /root/datasets/officeqa
# start Ray on all eight nodes, then on the head node:
DATABRICKS_HOST=<workspace-url> DATABRICKS_TOKEN=<token> MILES_SCRIPT_EXTERNAL_RAY=1 \
  python examples/air_officeqa/run_officeqa.py train --num-nodes 8 --save-hf
```

The judge still calls the workspace's serving endpoints, so the workers need `DATABRICKS_HOST` and
`DATABRICKS_TOKEN`. Without `--persist-dir`, saves stay in the launcher's output directory, and nothing is
copied or resumed.

## What's in this directory

| Path | What it is |
| --- | --- |
| `agent.py` | The agent loop: model turns, tool calls, and the final report |
| `reward.py` | The reward entry point, which scores the report |
| `prepare_data.py` | Downloads OfficeQA and builds the prompt files and the corpus |
| `corpus.py` | The bulletins, their search passages, and the four retrieval tools |
| `sandbox.py` | The Python sandbox, run with bubblewrap |
| `path_reward.py`, `path_report.py`, `judges.py`, `judges.json` | The answer and path checks, and the judge endpoints |
| `answer_check.py`, `answer_match.py` | The answer check, including OfficeQA's official scorer |
| `curriculum.py` | Samples questions by how often the policy solves them |
| `eval/` | The standalone SGLang eval that `eval.yaml` runs |
| `air/` | The three AIR jobs |
