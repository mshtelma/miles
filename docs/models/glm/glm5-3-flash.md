---
title: GLM-5.3-Flash
description: Agentic RL recipe for GLM-5.3-Flash, a KDA + DSA hybrid MoE, on OfficeQA at 150K tokens.
---

This recipe trains GLM-5.3-Flash as a tool-using agent on [OfficeQA](https://huggingface.co/datasets/databricks/officeqa).
OfficeQA asks questions about U.S. Treasury Bulletins. The agent searches and reads the bulletins,
computes in a Python sandbox, and submits an answer with the path of evidence behind it. Training is
fully async DAPO with a 150K-token context.

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

`scripts/run_glm5_3_flash.py` trains on eight 8-GPU nodes: four train with TP1 CP8 PP4 (11/11/11/12
layers) EP8, and the others serve the policy as 8-GPU SGLang engines, fully async. More nodes add
engines.

## 3. Environment Setup

The recipe runs on the Databricks AI Runtime image built from `docker/air/Dockerfile`
(`python docker/build.py --variant air-glm53`). It pins:

- miles;
- SGLang `sglang-miles-glm53next` at `9a26e749`, with the patch in `docker/air/patches`;
- Megatron-LM [`#89`](https://github.com/radixark/Megatron-LM/pull/89) at `e8f57451`;
- bubblewrap and bm25s, for the agent's tools.

Download and convert the checkpoint (`--ref-load` resolves to `<--ckpt-dir>/glm5.3-flash_torch_dist`):

```bash
python scripts/run_glm5_3_flash.py prepare                      # 1 node
```

Build the OfficeQA prompt files and corpus. The script downloads the gated `databricks/officeqa` dataset
at a pinned revision, so accept its terms on Hugging Face and export a token first. It keeps the
bulletins as the dataset's own text rendering, and cuts them into the passages the agent's search ranks:

```bash
HF_TOKEN=<token> python -m examples.glm5_3_flash.officeqa.prepare_data --out-dir /root/datasets/officeqa
```

The script trains on all 113 easy questions and 90 of the 133 hard ones, and holds out the other 43
hard questions in `eval_heldout.jsonl`.

## 4. Launch

On an 8-node Ray cluster (`MILES_SCRIPT_EXTERNAL_RAY=1`), from the head node:

```bash
python scripts/run_glm5_3_flash.py train --save-hf
```

The defaults are the recipe:

- 100 steps of 8 prompts x 8 samples, at 150K tokens per episode and 16K tokens per model turn;
- up to 128 agent turns;
- weight staleness up to 16 and 480 episodes in flight;
- routing replay, and Dr. GRPO (no reward-std normalization);
- a save every 10 steps.

The reward credits a correct answer only if an LLM judge confirms the cited path supports it. The
judge is a Databricks serving endpoint, listed in `examples/glm5_3_flash/officeqa/judges.json`, so
the workers need `DATABRICKS_HOST` and `DATABRICKS_TOKEN`.

`--save-hf` writes a drop-in HF checkpoint inside each save (`iter_*/hf`, in the base model's layout).
The trainer waits for it: in the 8-node recipe, a save with the export takes
10 to 12 minutes. Without `--save-hf`, convert a save later with `tools/convert_torch_dist_to_hf.py`.

On Databricks AI Runtime, three jobs run the whole flow: prepare, train and evaluate. See
[GLM-5.3-Flash on OfficeQA](/examples/glm5-3-flash).

## 5. Evaluation

`examples/glm5_3_flash/officeqa/air/eval.yaml` scores the base model or a `--save-hf` checkpoint on
the held-out questions: a standalone SGLang server set up like the rollout engines, running the
training agent, tools and reward. See
[GLM-5.3-Flash on OfficeQA](/examples/glm5-3-flash#3-evaluate).

## 6. What a Healthy Run Looks Like

A 50-step OfficeQA run of this recipe on 8x B300 showed these metrics:

| Metric | Observed |
|---|---|
| `train/train_rollout_logprob_abs_diff` | 0.03 – 0.05 |
| `train/grad_norm` | 0.06 – 0.16 |
| weight staleness | 9 – 14 (limit 16) |
| truncated episodes | 0 – 9% |
| episode length | 30K – 59K tokens |

`train/train_rollout_logprob_abs_diff` is the one to read first on a fresh bring-up. It covers the
KDA, DSA and hyper-connection paths at once.
