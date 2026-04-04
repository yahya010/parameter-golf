# Parameter Golf — Agent Onboarding Guide

You are helping yahya010 compete in the **Parameter Golf** competition (github.com/openai/parameter-golf). This file is your complete context.

## Competition Rules

- Train a GPT-2-style LM on FineWeb-10B (1024-token BPE vocab)
- **16 MB** max artifact (lzma-compressed weights + eval code)
- **600s** max training on **8×H100 SXM** (DDP, torchrun)
- **600s** max evaluation on same hardware
- Metric: **val_bpb** (bits-per-byte) — lower is better
- Must beat merged SOTA by **0.005 nats** minimum
- Submissions go in `records/track_10min_16mb/` as PRs to `openai/parameter-golf`

## Current Status (2026-04-03)

- **Our best:** val_bpb = **1.1393** (v21, seed 1337, single seed)
- **Merged SOTA:** val_bpb = **1.1147** (PR #1019, 3-seed mean, 1.8822 nats)
- **Best pending:** val_bpb = **0.8637** (PR #1313, SLOT-24, 3-seed mean)
- **Our next script:** `train_gpt_v25.py` — PR #1019 base + SLOT eval
- **Our PR:** #150 (DRAFT, old) on branch `submission/v12-next`

## Repository Layout

```
origin   https://github.com/yahya010/parameter-golf.git   (our fork)
upstream https://github.com/openai/parameter-golf.git      (competition repo)
```

**Key files:**
- `train_gpt_v25.py` — **Next submission script** (PR #1019 base + SLOT, UNTESTED on GPU)
- `train_gpt.py` — Upstream baseline (don't modify directly)
- `efforts.md` — Full optimization history and techniques reference
- `records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py` — Merged SOTA (PR #1019) source

## How to Run

### 8×H100 (RunPod, Nebius, etc.) — for real scoring

```bash
# Standard run (flash-attn required)
torchrun --standalone --nproc_per_node=8 train_gpt_v25.py

# With seed
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt_v25.py

# Without flash-attn (falls back to SDPA, slightly slower)
USE_FA3=0 torchrun --standalone --nproc_per_node=8 train_gpt_v25.py
```

### Local MacBook (for code iteration only)

```bash
pip install torch numpy sentencepiece zstandard tqdm

# Syntax + model init test (no data needed)
USE_FA3=0 python -c "
import os; os.environ['USE_FA3']='0'
import torch, sys; sys.path.insert(0,'.')
from train_gpt_v25 import GPT, Hyperparameters
args = Hyperparameters()
m = GPT(vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
    num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
    tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
    logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
    bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    xsa_last_n=args.xsa_last_n, rope_dims=args.rope_dims, ln_scale=args.ln_scale,
    ve_enabled=args.ve_enabled, ve_dim=args.ve_dim, ve_layers=args.ve_layers).float()
x = torch.randint(0, 1024, (2, 64))
with torch.no_grad(): h = m.forward_hidden(x)
print(f'OK: {sum(p.numel() for p in m.parameters()):,} params, hidden={h.shape}')
"
```

**Real BPB numbers MUST come from 8×H100 runs.**

## Submission Process

1. **Run 3 seeds on 8×H100:**
   ```bash
   for SEED in 1337 42 2025; do
     SEED=$SEED torchrun --standalone --nproc_per_node=8 train_gpt_v25.py
     cp final_model.int6.ptz final_model_seed${SEED}.int6.ptz
   done
   ```

2. **Create record folder:**
   ```bash
   mkdir -p records/track_10min_16mb/2026-04-XX_SLOT_GPTQ_XSA/
   cp train_gpt_v25.py records/track_10min_16mb/2026-04-XX_SLOT_GPTQ_XSA/train_gpt.py
   # Write README.md with results table
   ```

3. **Create a NEW branch and PR:**
   ```bash
   git checkout -b submission/v25-slot-gptq
   git add records/track_10min_16mb/2026-04-XX_SLOT_GPTQ_XSA/
   git push origin submission/v25-slot-gptq
   gh pr create --title "Record: SLOT + Full GPTQ + XSA-all — val_bpb X.XXXX (3-seed mean)" \
     --body "..." --repo openai/parameter-golf
   ```

## Architecture (v25 — PR #1019 base + SLOT)

| Component | Value |
|-----------|-------|
| Layers | 11, dim=512, 8 heads, 4 KV heads (GQA) |
| MLP | 3× expansion (hidden=1536), LeakyReLU(0.5)² |
| Embeddings | Tied, FP16 passthrough |
| Position | Partial RoPE (16/64 dims), base=10000 |
| SmearGate | Sigmoid blend current/previous token |
| BigramHash | 3072 buckets, dim=112 |
| OrthoInit | Orthogonal init + proj scaling |
| U-Net skips | 5 enc + 6 dec with learnable skip weights |
| XSA | All 11 layers |
| Value Embed | VE128 on layers 9,10 |
| LN Scale | 1/sqrt(layer+1) |
| QK Gain | Learnable per-head (init=1.5) |
| Optimizer | Parallel Muon (lr=0.025, mom=0.99, WD=0.04) |
| EMA | decay=0.997, applied at end |
| SWA | Every 50 steps when LR scale < 0.2 |
| Late QAT | Int6 STE when LR scale < 0.15 |
| Export | Full Hessian GPTQ int6 + LZMA-9 |
| GPTQ calib | AR self-gen (64 seqs, temp=0.8) |
| **SLOT** | **16 AdamW steps, lr=0.008→0.0008, per-sample delta + logit_bias** |
| Eval | Sliding window stride=64 + SLOT |

## Competition Landscape (2026-04-03)

| PR | Author | BPB | Key Technique |
|----|--------|-----|---------------|
| #1313 | anthony-maio | **0.8637** | SLOT-24 (eval-time only, 24 steps) |
| #1303 | anthony-maio | 0.9462 | SLOT + QK-Gain 4.0 + XSA-11 |
| #1229 | resouer | 0.9300 | SLOT-16 + scored-position masking |
| #1318 | renqianluo | 1.0096 | TTT-AdamW + SLOT + L-BFGS |
| #1289 | MatoTeziTanka | 1.0819 | Parallel Residuals + Depth Recurrence |
| #1306 | resouer | 1.0846 | Causal SLOT + Pre-quant TTT |
| #1296 | aryanbhosale | 1.0897 | Depth Recurrence + MuonEq-R |
| #1302 | vlivashkin | 1.1079 | Split-LR + N-gram Agreement + GPTQ |
| #1176 | bigbag | 1.0914 | SLOT-8 + QK-Gain 4.0 + XSA-all |
| **#1019** | **abaybektursun** | **1.1147** | **Merged SOTA: AR GPTQ + XSA-all** |

### Key Technique: SLOT (Self-Learned Online Test-time optimization)

SLOT (arXiv:2505.12392v2) is the biggest advancement since our last session:
- **Pure eval-time** — doesn't change training at all
- Model weights **frozen**; only per-window throwaway delta + logit_bias optimized
- Per-sample delta [bsz, 1, 512] + logit_bias [bsz, 1, 1024]
- 16-24 AdamW steps with cosine LR schedule
- Scored-position masking (only last `stride` tokens per window)
- Gain: **0.02–0.26 BPB** depending on steps/LR
- Legal: accepted in PRs #1176, #1229, #1313

## Critical Lessons

### SLOT is the Game-Changer
PR #1313 goes from 1.1229 sliding BPB to 0.8637 with SLOT-24. That's 0.26 BPB from eval-time alone. Even conservative 8-step SLOT gives ~0.02 BPB (PR #1176).

### Merged SOTA Uses Full Hessian GPTQ (not simple Int6)
PR #1019 uses Cholesky error compensation + column reordering. Quant gap dropped from ~0.04 to ~0.002 BPB.

### AR Self-Gen Calibration is Legal
Model generates its own 64×2048 calibration tokens (temp=0.8). No training/val data accessed during quantization.

### LZMA > zstd for Compression
PR #1019 switched from zstd to LZMA preset=9 for ~5% better compression.

### SWA + QAT = Broken
SWA averages checkpoints, destroying QAT robustness. Use Late QAT with EMA→SWA stacking (SOTA approach).

### Weight Decay = Compression Hack
WD=0.04 shrinks weights → better compression → more parameters under 16MB.

### Submission Rules
- Beat merged SOTA by **0.005 nats** minimum
- 3-seed results required
- Create NEW PRs (don't edit existing ones)
- Score-first TTT/SLOT is legal (tokens scored before weight update)

## Environment Vars (v25)

```bash
# Architecture
NUM_LAYERS=11  MODEL_DIM=512  NUM_HEADS=8  NUM_KV_HEADS=4
BIGRAM_VOCAB_SIZE=3072  BIGRAM_DIM=112  ROPE_DIMS=16
XSA_LAST_N=11  LN_SCALE=1  VE_ENABLED=1  VE_DIM=128
LOGIT_SOFTCAP=30  QK_GAIN_INIT=1.5

# Training
TRAIN_SEQ_LEN=2048  TRAIN_BATCH_TOKENS=786432  ITERATIONS=20000
MAX_WALLCLOCK_SECONDS=600  WARMDOWN_ITERS=4000  WARMUP_STEPS=20

# Optimizer
MATRIX_LR=0.025  MUON_MOMENTUM=0.99  MUON_WD=0.04

# EMA + SWA
SWA_ENABLED=1  SWA_EVERY=50  # EMA decay=0.997 hardcoded

# Late QAT
LATE_QAT_THRESHOLD=0.15

# SLOT
SLOT_ENABLED=1  SLOT_STEPS=16  SLOT_LR=0.008  SLOT_LR_MIN=0.0008

# Eval
EVAL_STRIDE=64  VAL_BATCH_SIZE=524288

# Flash Attention
USE_FA3=1  # Set to 0 if flash-attn not available
```

## GPU Run Checklist

When you get H100 access:
1. Install: `pip install torch flash-attn sentencepiece lzma numpy`
2. Download data: `python data/download.py`
3. Single-seed test: `SEED=1337 torchrun --standalone --nproc_per_node=8 train_gpt_v25.py`
4. Check output for `final_slot` line — that's the SLOT BPB
5. If BPB < 1.11: run 3 seeds and submit PR
