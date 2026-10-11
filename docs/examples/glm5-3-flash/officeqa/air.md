---
title: "OfficeQA RL with GLM-5.3-Flash on Databricks AI Runtime"
description: "Agentic RL of GLM-5.3-Flash on OfficeQA at 150K tokens on Databricks AI Runtime, from the published model to an evaluated checkpoint in four jobs."
# Generated from examples/glm5_3_flash/officeqa/air/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
Three AIR jobs take you from the published model to an evaluated checkpoint. Run them in order:

| Job | Nodes (8x B300) | What it does | What it leaves in `VOLUME_ROOT` |
| --- | --- | --- | --- |
| `prepare.yaml` | 1 | Downloads GLM-5.3-Flash from Hugging Face, converts it to the Megatron `torch_dist` reference the trainer loads, and builds the OfficeQA prompt files. | `models/GLM-5.3-Flash/`, `ckpt/glm5.3-flash_torch_dist/`, `data/officeqa/` |
| `train.yaml` | 8 | Fully async DAPO on OfficeQA at 150K tokens: four nodes train, four serve the policy with SGLang. | `runs/<RUN_ID>/checkpoints/` (Megatron saves, each with a drop-in HF checkpoint under `iter_<step>/hf`) and `runs/<RUN_ID>/logs/` |
| `eval.yaml` | 1 | Serves an HF checkpoint with standalone SGLang and scores the held-out questions with the training agent, tools and reward. | `runs/<EVAL_RUN_ID>/<attempt>/` |

A job's attempt is its AIR run's `MLFLOW_RUN_ID`.

## Before you start

* **A Volume directory** for everything the jobs write, `VOLUME_ROOT` below.
* **The OfficeQA data** on a Volume:
  * `officeqa_full.csv`, from the gated [`databricks/officeqa`](https://huggingface.co/datasets/databricks/officeqa) dataset. Accept its terms on Hugging Face first.
  * The Treasury Bulletin corpus the agent's tools search: `treasury_bulletins_clean.zip` and its BM25 passages `chunks.jsonl`.
* **The runtime image.** Build and push `docker/air/Dockerfile`, then register the digest the push reports:

  ```bash
  python docker/build.py --variant air-glm53 --image-tag custom --custom-tag glm53 \
    --image <registry>/miles-air --push
  air -p <profile> register image <registry>/miles-air@sha256:<digest>
  ```

  Where the build can't reach PyPI, add `--build-arg PIP_INDEX_URL=<mirror>`.
* **The judge.** The reward's LLM judge calls the Databricks serving endpoints listed in
  `../judges.json`, with the workspace credentials the AIR node already carries.

Every command below runs from the repository root. It starts with these overrides:

```bash
COMMON="environment.docker_image.url=<registry>/miles-air@sha256:<digest> \
  env_variables.VOLUME_ROOT=/Volumes/<catalog>/<schema>/<volume>/miles"
```

For a reproducible, cached snapshot of the code, add `code_source.snapshot.git.commit=$(git rev-parse HEAD)`.
Without it, AIR uploads the working tree, including uncommitted changes.

## 1. Prepare the inputs (about 40 minutes)

```bash
air -p <profile> run examples/glm5_3_flash/officeqa/air/prepare.yaml --override $COMMON \
  env_variables.QUESTIONS_CSV=/Volumes/<catalog>/<schema>/<volume>/officeqa/officeqa_full.csv \
  env_variables.CORPUS_DIR=/Volumes/<catalog>/<schema>/<volume>/officeqa
```

`scripts/run_glm5_3_flash.py prepare` downloads the pinned Hugging Face revision and converts it with
`tools/convert_hf_to_torch_dist.py` under `torchrun`. `examples/glm5_3_flash/officeqa/prepare_data.py`
splits the questions: all 113 easy questions and 90 of the 133 hard ones go into `train.jsonl`, and the
other 43 hard questions into `eval_heldout.jsonl`. It also writes `curriculum.json` and copies the
corpus. `tools/air/copy_tree.py` then copies all three to the Volume. The reference's
`latest_checkpointed_iteration.txt` goes last, so a partial copy fails to load instead of loading half.

## 2. Train

```bash
air -p <profile> run examples/glm5_3_flash/officeqa/air/train.yaml --override $COMMON
```

`TRAIN_ARGS` is passed to `scripts/run_glm5_3_flash.py train`; see that script's docstring for the
flags. The default, `--save-hf --mlflow-experiment glm53-flash-officeqa`, trains 100 steps and saves
every 10. Each save includes an HF checkpoint for `eval.yaml`, and Miles logs into the AIR job's MLflow
run.

With `--save-hf`, each save holds the trainer 10 to 12 minutes while it writes the 328 GB HF
checkpoint. Without it, only the newest save can be converted later, with
`tools/convert_torch_dist_to_hf.py`.

For a short check, set
`env_variables.TRAIN_ARGS="--save-hf --num-rollout 2 --save-interval 2 --mlflow-experiment glm53-flash-officeqa"`.
Setting `TRAIN_ARGS` replaces the default, so keep `--mlflow-experiment`, or Miles logs no metrics.

What the job does:

* **Staging.** Every node copies the HF checkpoint and the OfficeQA data to its local disk, then
  joins Ray; the head node runs the launcher.
* **Saves.** Megatron saves to each node's local disk. Once a save's trainer and rollout state are
  written, `tools/air/publisher.py` copies every node's part of it to `runs/<RUN_ID>/checkpoints/` with
  plain file copies and deletes it locally; training goes on meanwhile. Megatron's tracker,
  `latest_checkpointed_iteration.txt`, is written last. The older saves then lose their training
  state and keep only their HF checkpoint. The next save waits for the copy, and a failed copy fails
  the run.
* **Resume.** Submit again with the same `RUN_ID`. Miles loads the save the tracker names, straight
  from the Volume, or starts from the converted reference when there is none. A resume must keep
  `--num-rollout`, because Megatron rejects a different LR schedule length.
* **Logs.** `air logs` keeps only the start of a job's output, so the launcher's log is copied to
  `runs/<RUN_ID>/logs/<attempt>.log` every minute.

## 3. Evaluate

```bash
# the base model
air -p <profile> run examples/glm5_3_flash/officeqa/air/eval.yaml --override $COMMON
# a training save
air -p <profile> run examples/glm5_3_flash/officeqa/air/eval.yaml --override $COMMON \
  env_variables.HF_SOURCE=/Volumes/<catalog>/<schema>/<volume>/miles/runs/<RUN_ID>/checkpoints/iter_<step>
```

A save qualifies once its copy has finished (the tracker has reached it) and its HF export completed
(`iter_<step>/hf/.complete`). `EVAL_ARGS` takes `examples/glm5_3_flash/officeqa/eval/entry.py` flags
such as `--samples-per-prompt 4` or `--limit-questions 3`.

Each eval writes `eval_results.json` (the rewards per question), `summary.json` (pass@k, questions
solved, lost trajectories), `trajectories.jsonl` and the SGLang server's log.

## Things to know about AIR

* `timeout_minutes` includes the time a job waits for GPUs.
* AIR stops a job when any of its nodes has had all GPUs idle for 60 minutes. Every step here stays
  below that, but a much larger dataset or a very slow first rollout can exceed it.
* A run keeps one 4.4 TB training state, its newest save's, plus a 328 GB HF checkpoint per save.
  Delete the runs you no longer need.
