# OfficeQA RL with GLM-5.3-Flash on Databricks AI Runtime

Four AIR jobs take you from the published model to an evaluated checkpoint. Run them in order:

| Job | Nodes (8x B300) | What it does | What it leaves in `VOLUME_ROOT` |
| --- | --- | --- | --- |
| `prepare.yaml` | 1 | Builds the OfficeQA prompt files, downloads GLM-5.3-Flash from Hugging Face, and publishes both. | `models/`, `datasets/`, the input manifest `inputs/<sha256>.json`, and `inputs/latest` |
| `convert.yaml` | 1 | Converts the HF checkpoint to the Megatron `torch_dist` reference the trainer loads. | `ckpt/glm5.3-flash_torch_dist_<attempt>/` and `ckpt/latest` |
| `train.yaml` | 8 | Fully async DAPO on OfficeQA at 150K tokens: four nodes train, four serve the policy with SGLang. | `runs/<RUN_ID>/checkpoints/<attempt>/iter_<step>/`, each save with a drop-in HF checkpoint under `iter_<step>/hf` |
| `eval.yaml` | 1 | Serves an HF checkpoint with standalone SGLang and scores the held-out questions with the training agent, tools and reward. | `runs/<EVAL_RUN_ID>/attempts/<attempt>/eval/` |

Each job finds the previous job's output through `inputs/latest` and `ckpt/latest`, so the defaults
chain. To pin other inputs, set `INPUT_MANIFEST` or `REFERENCE_ROOT` instead. A job's attempt is its
AIR run's `MLFLOW_RUN_ID`.

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

## 1. Prepare the inputs (about 25 minutes)

```bash
air -p <profile> run examples/glm5_3_flash/officeqa/air/prepare.yaml --override $COMMON \
  env_variables.QUESTIONS_CSV=/Volumes/<catalog>/<schema>/<volume>/officeqa/officeqa_full.csv \
  env_variables.CORPUS_DIR=/Volumes/<catalog>/<schema>/<volume>/officeqa
```

`examples/glm5_3_flash/officeqa/prepare_data.py` splits the questions: all 113 easy questions and 90
of the 133 hard ones go into `train.jsonl`, and the other 43 hard questions into `eval_heldout.jsonl`.
It also writes `curriculum.json` and copies the corpus. `tools/air/stage_models.py prepare` publishes
that directory and the HF checkpoint pinned in `inputs.json`. The input manifest lists every file's
size and SHA-256, and the later jobs verify each file they stage against it.

## 2. Convert the reference (about 20 minutes)

```bash
air -p <profile> run examples/glm5_3_flash/officeqa/air/convert.yaml --override $COMMON
```

This stages the HF checkpoint onto the node and runs `scripts/run_glm5_3_flash.py prepare --skip-download`,
which runs `tools/convert_hf_to_torch_dist.py` under `torchrun`. `stage_models.py publish-reference`
then copies the result to the Volume and writes its `COMPLETE.json` last, so the trainer never loads
a partial reference.

## 3. Train

```bash
air -p <profile> run examples/glm5_3_flash/officeqa/air/train.yaml --override $COMMON
```

`TRAIN_ARGS` is passed to `scripts/run_glm5_3_flash.py train`; see that script's docstring for the
flags. The default, `--save-hf --mlflow-experiment glm53-flash-officeqa`, trains 100 steps and saves
every 10. Each save includes an HF checkpoint for `eval.yaml`, and Miles logs into the AIR job's MLflow
run.

With `--save-hf`, each save holds the trainer 10 to 12 minutes while it writes the 328 GB HF
checkpoint and checks it against the base model. Without it, convert a save later with
`tools/convert_torch_dist_to_hf.py`.

For a short check, set
`env_variables.TRAIN_ARGS="--save-hf --num-rollout 2 --save-interval 2 --mlflow-experiment glm53-flash-officeqa"`.
Setting `TRAIN_ARGS` replaces the default, so keep `--mlflow-experiment`, or Miles logs no metrics.

To resume, submit again with the same `RUN_ID`. The job continues from the newest complete save of
any attempt of that run. A resume must keep `--num-rollout`, because Megatron rejects a different LR
schedule length. It also refuses a save trained with a different model, tokenizer, chat template,
data, optimizer or parallel layout. Both checks run before Ray starts.

What the job does (`tools/air/node_entry.py`, one supervisor per node):

* **Staging.** The head node picks the save to resume, starts Ray and, once every node has joined,
  stages onto each node's local disk the HF weights, the prompt data and the small checkpoint files
  (tracker, metadata, common state, rollout cursor). Each trainer rank then fetches only the
  checkpoint shards its read plan needs (`tools/air/dcp_stage.py`), once per node.
* **Launch.** The head runs the launcher command that `recipe.py` builds; the other nodes wait for
  its outcome.
* **Saves.** Trainer ranks save to local disk. After each save, `tools/air/publisher.py` copies every
  node's files to `runs/<RUN_ID>/checkpoints/<attempt>/iter_<step>/`, verifies them, and writes
  `COMPLETE.json` last; a resume reads only saves that have it. Training goes on during the copy,
  the next save waits for it, and a failed copy fails the run. Each node keeps its two newest saves.
* **Logs.** `air logs` keeps only the start of a job's output. The launcher's log is published every
  30 seconds, as numbered chunks under `runs/<RUN_ID>/attempts/<attempt>/logs/launcher/`.

## 4. Evaluate

```bash
# the base model
air -p <profile> run examples/glm5_3_flash/officeqa/air/eval.yaml --override $COMMON
# a training save
air -p <profile> run examples/glm5_3_flash/officeqa/air/eval.yaml --override $COMMON \
  env_variables.HF_SOURCE=checkpoint:/Volumes/<catalog>/<schema>/<volume>/miles/runs/<RUN_ID>/checkpoints/<attempt>/iter_<step>
```

`HF_SOURCE=checkpoint:` accepts only complete saves whose HF export passed its checks
(`iter_<step>/hf_export.json`). `EVAL_ARGS` takes `examples/glm5_3_flash/officeqa/eval/entry.py` flags
such as `--samples-per-prompt 4` or `--limit-questions 3`.

Each eval writes `eval_results.json` (the rewards per question), `summary.json` (pass@k, questions
solved, lost trajectories), `trajectories.jsonl` and the SGLang server's log.

## Things to know about AIR

* `timeout_minutes` includes the time a job waits for GPUs.
* AIR stops a job when any of its nodes has had all GPUs idle for 60 minutes. Every step here stays
  below that, but a much larger dataset or a very slow first rollout can exceed it.
