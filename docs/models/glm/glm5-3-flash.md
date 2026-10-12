---
title: GLM-5.3-Flash
description: GLM-5.3-Flash, a KDA + DSA hybrid MoE, with fully async DAPO on math, and as an agent on OfficeQA at 150K tokens.
---

Two recipes train GLM-5.3-Flash on the same eight-node layout. `scripts/run_glm5_3_flash.py` is its standard
Miles recipe: fully async DAPO on DAPO-Math-17k, and any other task by flags. The
[OfficeQA example](/examples/air-officeqa) trains it as a tool-using research agent at 150K tokens per episode.

## 1. Model Introduction

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (`model_type: glm5_next`) is a
**45-layer KDA + DSA hybrid MoE**. It is a different architecture from the 744 B GLM5 and
GLM5.2 flagships, not a smaller cut of them.

- **45 layers, hybrid**: 34 KDA linear-attention layers + 11 DSA sparse-attention layers.
- **288-expert MoE**, sigmoid routing at top-8; the first 3 layers are dense.
- **mHC hyper-connections** at every block.
- **NoPE MLA** — multi-latent attention with the positional half of the QK head empty.
- **kpool-compressed lightning indexer** picks which keys the DSA layers attend.
- Hidden 4096, FFN 12288, 64 attention heads, vocab 154880.
- MTP is dropped for training; exported checkpoints carry the base model's MTP layer unchanged.

## 2. Layout

Both recipes train on eight 8-GPU nodes: four train with TP1 CP8 PP4 (11/11/11/12 layers) EP8, and the others
serve the policy as 8-GPU SGLang engines, fully async. More nodes add engines.

## 3. Environment Setup

Both recipes run on the Databricks AI Runtime image built from `docker/air/Dockerfile`
(`python docker/build.py --variant air-glm53`). It pins:

- miles;
- SGLang `sglang-miles-glm53next` at `9a26e749`, with the patch in `docker/air/patches`;
- Megatron-LM [`#89`](https://github.com/radixark/Megatron-LM/pull/89) at `e8f57451`;
- bubblewrap and bm25s, for the agent's tools.

Download the checkpoint and the math datasets, and convert the checkpoint (`--ref-load` resolves to
`<--ckpt-dir>/glm5.3-flash_torch_dist`):

```bash
python scripts/run_glm5_3_flash.py prepare                      # 1 node
```

## 4. Launch

On an 8-node Ray cluster (`MILES_SCRIPT_EXTERNAL_RAY=1`), from the head node:

```bash
python scripts/run_glm5_3_flash.py train --save-hf
```

The defaults are the standard math recipe:

- 3000 steps of 32 prompts x 8 samples, at 32K tokens per sample and 8K per response, with the deepscaler reward;
- AIME-2024 every 20 steps, 16 samples each;
- weight staleness up to 16 and 512 samples in flight;
- routing replay, with in-flight requests paused in place for each weight update;
- a save every 20 steps.

Another task is a matter of flags: `--prompt-data` for the prompts, `--no-enable-eval`, and `--extra-args` for
the rest, such as `--custom-rm-path`. `--save-hf` writes a drop-in HF checkpoint inside each save (`iter_*/hf`,
in the base model's layout); the trainer waits for it, about 10 minutes a save. Without `--save-hf`, convert a
save later with `tools/convert_torch_dist_to_hf.py`.

A four-step check of these defaults on eight 8x B300 nodes showed:

| Metric | Observed |
|---|---|
| `train/train_rollout_logprob_abs_diff` | 0.03 |
| `train/grad_norm` | 0.18 – 0.24 |
| `eval/aime24` | 0.77 – 0.80 |
| response length | 4.7K – 5.7K tokens, 29 – 42% of them at the 8K limit |

The first step lands about 75 minutes after the job starts: the engines load, AIME-2024 is evaluated once,
and the first rollout fills. Later steps take a few minutes each.

On Databricks AI Runtime, `tools/air/prepare.yaml` and `tools/air/train.yaml` run this launcher (see
[Miles on Databricks AI Runtime](https://github.com/radixark/miles/blob/main/tools/air/README.md)).

## 5. Agentic RL on OfficeQA

[OfficeQA](https://huggingface.co/datasets/databricks/officeqa) asks questions about U.S. Treasury Bulletins.
The agent searches and reads the bulletins, computes in a Python sandbox, and submits an answer with the path
of evidence behind it. `examples/air_officeqa/run_officeqa.py` trains it on the layout above, fully async at
150K tokens per episode:

- 100 steps of 8 prompts x 8 samples, at 16K tokens per model turn and up to 128 agent turns;
- weight staleness up to 16 and 480 episodes in flight;
- routing replay through the session server, and Dr. GRPO (no reward-std normalization);
- a save every 10 steps.

The reward credits a correct answer only if an LLM judge confirms the cited path supports it. The judge is a
Databricks serving endpoint, listed in `examples/air_officeqa/judges.json`, so the workers need
`DATABRICKS_HOST` and `DATABRICKS_TOKEN`. The example also has the data preparation, a standalone eval of the
base model or a save on 43 held-out questions, and the AIR jobs for all three: see
[OfficeQA on Databricks AI Runtime](/examples/air-officeqa).

## 6. What a Healthy OfficeQA Run Looks Like

A 50-step OfficeQA run of this recipe on eight 8x B300 nodes showed these metrics:

| Metric | Observed |
|---|---|
| `train/train_rollout_logprob_abs_diff` | 0.03 – 0.05 |
| `train/grad_norm` | 0.06 – 0.16 |
| weight staleness | 9 – 14 (limit 16) |
| truncated episodes | 0 – 9% |
| episode length | 30K – 59K tokens |

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up. It covers the
KDA, DSA and hyper-connection paths at once.

A longer run, evaluated on the 43 held-out questions every 50 steps, shows where the gains come from:

| Steps trained | 0 | 50 | 100 | 150 | 175 |
|---|---|---|---|---|---|
| pass@1 | 0.39 | 0.47 | 0.47 | 0.48 | 0.47 |
| pass@3 | 0.54 | 0.63 | 0.61 | 0.62 | 0.62 |
| fabricated answers | 11.6% | 4.1% | 1.8% | 2.3% | 0.6% |

Held-out pass@1 gains about eight points in the first 50 steps and stays flat after that, while the training
reward keeps rising, to 0.86 at step 175: with 203 training questions, later steps mostly fit those. Fabricated
answers fall from 12% to under 1%. Evaluate saves as the run goes, and keep the one where held-out accuracy
stops improving.
