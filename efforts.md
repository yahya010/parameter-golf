# Parameter Golf — Optimization Efforts Report

**Author:** yahya010
**Last Updated:** 2026-03-22
**Best Score:** val_bpb = **1.1393** (v21, seed 1337)
**Current SOTA:** val_bpb = **1.1027** (PR #442, AdamW TTT, 3-seed mean)
**Our Gap:** ~0.037 BPB

---

## Competition Overview

**Parameter Golf** is a language model compression competition. The goal: train the best small language model that:

- Fits in a **16 MB artifact** (zstd-compressed weights + code)
- Trains in **≤ 600 seconds** on **8×H100 SXM GPUs**
- Evaluates in **≤ 600 seconds** on the same hardware
- Achieves the lowest **bits-per-byte (BPB)** on a held-out FineWeb validation set

The base model is a GPT-2-style transformer with 1024-token BPE vocabulary, trained on FineWeb-10B.

---

## Environment Setup

### Prerequisites

```bash
# Python 3.10+, CUDA 12.x, PyTorch 2.8+
conda create -n pgolf python=3.10
conda activate pgolf

# Core dependencies
pip install torch numpy sentencepiece zstandard tqdm

# Flash Attention (optional, for FA3)
pip install flash-attn --no-build-isolation

# Modal (for cloud 8×H100 training)
pip install modal
modal setup  # one-time auth
```

### Clone and Prepare Data

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Download the FineWeb-10B dataset (preprocessed)
python data/download.py

# Verify data structure:
# data/datasets/fineweb10B_sp1024/fineweb_train_*.bin
# data/datasets/fineweb10B_sp1024/fineweb_val_*.bin
# data/tokenizers/fineweb_1024_bpe.model
```

### Modal Setup (Cloud 8×H100)

```bash
# Create persistent volume for data
modal volume create pg-data

# Upload dataset and tokenizer
modal volume put pg-data ./data/datasets/fineweb10B_sp1024/ /datasets/fineweb10B_sp1024/
modal volume put pg-data ./data/tokenizers/ /tokenizers/

# Run training
modal run run_modal.py --script train_gpt_v21.py
modal run run_modal.py --script train_gpt_v21.py --seed 42
modal run run_modal.py --script train_gpt_v21.py --env "NUM_LAYERS=11,MUON_WD=0.04"
```

### Local MacBook Pro (Apple Silicon — for code iteration only)

```bash
pip install torch numpy sentencepiece zstandard tqdm

# Sanity test — NOT for BPB scoring (different hardware = different step count)
USE_FA3=0 MAX_WALLCLOCK_SECONDS=30 ITERATIONS=50 python train_gpt_v21.py
```

**What works on MacBook:** syntax checking, crash testing, export/quantization pipeline testing (does artifact fit 16MB?), new architecture code path validation.

**What doesn't work:** Flash Attention 3 (CUDA only), `torch.compile` CUDA backends, DDP, meaningful BPB numbers. Step throughput is completely different, so you'll get a different number of steps in 600s → meaningless loss.

**Workflow:** Iterate code on MacBook, only run `modal run run_modal.py --script ...` for real BPB numbers. Each Modal run costs ~$3-5 (10-15 min of 8×H100).

### Local Single-GPU (if 1×H100 available)

```bash
# Single-GPU, directionally accurate but not official scoring
python train_gpt_v21.py
```

---

## Best Configuration (v21) — 1.1393 BPB

### Architecture

| Component | Value |
|-----------|-------|
| Layers | 11 |
| Model dim | 512 |
| Heads | 8 (4 KV heads, GQA) |
| MLP | 3× expansion (hidden=1536), ReLU² |
| Vocab | 1024 BPE |
| Seq len | 2048 |
| Embeddings | Tied, FP16 passthrough |
| Position | NTK-RoPE (base=50000) |
| Logit cap | softcap=30 |
| SmearGate | Learned sigmoid blending with previous token |
| BigramHash | 2048 buckets, dim=128 |

### Training

| Parameter | Value |
|-----------|-------|
| Optimizer | Muon (lr=0.025, momentum=0.99, 5 N-S steps) |
| Weight Decay | 0.04 (on Muon params only) |
| Embed LR | 0.6 (Adam) |
| Head LR | 0.008 (Adam) |
| Tied Embed LR | 0.035 |
| Batch tokens | 786,432 per step |
| Warmup | 20 steps |
| Warmdown | 3000 iters (of 20000 budget) |
| Grad clip | 0.3 |
| OrthoInit | Orthogonal init + muP scaling |
| QAT | **Disabled** (SWA replaces it) |

### Export & Eval

| Feature | Value |
|---------|-------|
| Quantization | Int6 at export only (no QAT) |
| Compression | zstd level 22 |
| Artifact size | ~15.8 MB (under 16 MB) |
| SWA | Every 50 steps when LR scale < 0.5 |
| TTT | SGD (lr=0.002, momentum=0.9, 3 epochs, freeze 0 blocks) |
| Eval | Sliding window stride=64 |

### Result Breakdown

```
Training:     ~5166 steps in 600s (~116ms/step)
Train BPB:    ~1.10
Pre-quant:    ~1.13
Post-quant:   ~1.17 (sliding window → 1.1393)
TTT gain:     ~0.015
Sliding gain: ~0.02
```

---

## Full Optimization Timeline

### Phase 1: Baseline (v1–v3) — BPB ~1.25

- Started from the repo's `train_gpt.py` baseline (9L, 512 dim, seq=1024)
- v1: Added seq_len=2048 → ~0.02 improvement
- v2: FP16 tied embeddings (keep embedding in fp16 during quant) → ~0.01
- v3: Muon weight decay tuning

### Phase 2: Int6 Quantization (v4–v8) — BPB ~1.16

- v4: STE Int6 QAT (`fake_quantize_int6`) with straight-through estimator
  - Range: [-31, 31], 6-bit signed integer
  - Eliminated the quant gap entirely (pre-quant ≈ post-quant)
- v5: BigramHash embedding (2048 buckets, dim=128) — hash of token pairs as extra features
- v6: MLP expansion to 1344, tuned LRs → **1.1598 BPB** (10L best with QAT)
- v7–v8: Sweep attempts, minor gains

### Phase 3: 11 Layers + SWA (v9–v14) — BPB ~1.15

- **Key insight**: WD=0.04 on Muon shrinks weights → zstd compresses ~20% better → room for 11th layer with MLP=1536
- v9–v11: 11L configs with various QAT + SWA combos
- **Critical discovery: SWA destroys QAT robustness** — averaging quantization-trained checkpoints breaks the individual quant-surviving property. Must choose one or the other.
- v12: Dropped QAT, rely on WD+SWA for compression budget → significant improvement
- v14: Best 10L with WD=0.02 → 1.1577

### Phase 4: Advanced Techniques (v15–v21) — BPB 1.1393

- v15: SmearGate + OrthoInit integrated
- v16–v17: Flash Attention 3 integration (via `flash_attn.flash_attn_interface`)
  - Required `torch.compiler.allow_in_graph(flash_attn_func)` for fullgraph compile
  - FA3 is ~7ms slower than SDPA with torch.compile due to graph breaks
- v18: U-Net skip connections tested — incompatible with CUDA graphs, abandoned
- v19: SWA tuning (every 50 steps, trigger at scale < 0.5)
- v20: Export artifact size issues (amax vs percentile clipping)
- v21: **Final best** — no QAT + SWA + TTT(freeze=0) + all features → **1.1393 BPB**

### Phase 5: SOTA Investigation (v22–v25) — Not Yet Validated

- v24: Ported features from PR #315/#338 (current leader):
  - Partial RoPE (16 dims)
  - LN Scale (learned layer norm scaling)
  - Late QAT (enable QAT only in final warmdown phase)
  - XSA4 (cross-sequence attention with 4 past tokens)
  - EMA (exponential moving average of weights)
- v24 failed on Modal due to FA3 compile issues — needs SDPA fallback (`USE_FA3=0`)
- **These techniques remain untested and are the primary path to closing the gap**

---

## Techniques Deep Dive

### SWA (Stochastic Weight Averaging)

Average model checkpoints taken during the learning rate warmdown phase. We take a snapshot every 50 steps once the LR scale drops below 0.5.

**Why it works:** Averages out noise from SGD, finding a flatter minimum that generalizes better. Provides ~0.01–0.02 BPB gain.

**Why it conflicts with QAT:** Each QAT-trained checkpoint has individually learned to survive quantization. Averaging weights from different checkpoints produces a model that no individual checkpoint's quantization calibration covers — the averaged weights hit quant boundaries differently, destroying robustness.

**Our solution:** Drop QAT entirely. Use WD=0.04 + SWA + int6 export-only quantization. The quant gap is larger (~0.04) but overall BPB is lower because SWA's smoothing gain exceeds the quant gap penalty.

### TTT (Test-Time Training)

After export, run a few epochs of SGD on the validation data itself before scoring:

```python
# Pseudocode
model.load_quantized_weights()
optimizer = SGD(model.parameters(), lr=0.002, momentum=0.9)
for epoch in range(3):
    for batch in val_data:
        loss = model(batch).cross_entropy()
        loss.backward()
        optimizer.step()
# Then evaluate
```

**Gain:** ~0.015 BPB. The model adapts its weights to the specific distribution of the validation set.

**freeze_blocks parameter:** Originally froze first N blocks to prevent catastrophic forgetting, but freeze=0 (train all blocks) worked best.

### Sliding Window Evaluation

Instead of chunking validation data into non-overlapping seq_len windows:

```
Standard: [0:2048], [2048:4096], [4096:6144], ...
Sliding:  [0:2048], [64:2112], [128:2176], ...
          Score only the last `stride` tokens of each window
```

Each token gets scored with maximum context. **Gain:** ~0.02 BPB for free (eval-time only, no training change).

### Muon Optimizer

Newton-Schulz orthogonalization of gradients. Normalizes the gradient matrix to have orthogonal rows/columns, then applies momentum. Key hyperparameters:
- 5 Newton-Schulz steps (convergence iterations)
- Momentum 0.99 with warmup from 0.92 over 1500 steps
- Weight decay 0.04 (critical for compression)

### SmearGate

A learned sigmoid gate that blends the current token embedding with the previous token:

```python
gate = sigmoid(linear(x))
x = gate * x + (1 - gate) * x_prev
```

Gives the model a cheap "look back" at the prior token without attention.

### BigramHash

Hash embedding that encodes token-pair information:

```python
bigram_id = hash(prev_token, curr_token) % 2048
bigram_embed = embedding_table[bigram_id]  # dim=128
x = concat(token_embed, bigram_embed)  # projected down
```

### NTK-RoPE

Rotary position embedding with base frequency 50000 (vs default 10000). Better extrapolation to longer contexts. Combined with seq_len=2048.

### Int6 Quantization

Maps float weights to [-31, 31] integer range (6 bits):
```python
scale = max(abs(w)) / 31
w_int6 = round(w / scale).clamp(-31, 31)
w_reconstructed = w_int6 * scale
```

With zstd-22 compression, 6 bits per weight + scale overhead fits 11 layers under 16 MB.

---

## Key Lessons Learned

### 1. SWA and QAT Don't Mix
QAT trains each checkpoint to be individually robust to quantization. Averaging breaks this. Choose one.

### 2. Weight Decay is a Compression Hack
Higher WD → smaller weight magnitudes → lower entropy → better zstd compression ratio → more capacity for parameters under the 16 MB cap.

### 3. Step Throughput Matters Enormously
At 600s budget:
- 82ms/step = ~7317 steps (leader)
- 116ms/step = ~5172 steps (us)
That's 40% fewer gradient updates. Every ms of overhead costs ~7 training steps.

### 4. IEEE 754 NaN Behavior
- `NaN > 0` → FALSE, `NaN <= 0` → FALSE
- PyTorch's relu: `a <= 0 ? 0 : a` preserves NaN
- Must match exactly in custom CUDA kernels

### 5. torch.compile Interactions
- Flash Attention 3 requires `torch.compiler.allow_in_graph()` for fullgraph mode
- U-Net skip connections break CUDA graphs
- `max-autotune` mode can conflict with dynamic tensor storage

---

## Gap Analysis: 1.1393 → 1.1027 (Current SOTA)

Our best is **0.037 BPB** behind the leader (PR #442). The leaderboard has moved significantly.

### Current Top Entries (as of 2026-03-22)

| PR | Author | BPB | Key Technique |
|----|--------|-----|---------------|
| #442 | sjp611 | **1.1027** (mean) | 11L EMA + **AdamW TTT 10ep** |
| #445 | newjordan | 1.1232 | TTT Burst + EMA + GPTQ-lite |
| #401 | newjordan | 1.1243 | EMA + Tight SWA + Late QAT + VE128 + Partial RoPE |
| #452 | ofirkris | 1.1365 | XSA + EMA + Partial RoPE + LN Scale + TTT |
| #434 | parinzee | 1.1370 | XSA + LeakyReLU² + Partial RoPE |
| **#150** | **yahya010** | **1.1393** | **Our entry (v21, DRAFT)** |

### What the SOTA Does That We Don't

| Technique | SOTA (PR #442) | Us (v21) | Est. Gain |
|-----------|----------------|----------|-----------|
| **AdamW TTT** | lr=0.0005, 10 epochs | SGD lr=0.002, 3 epochs | **~0.019** |
| **EMA** (decay=0.997) | Stacked with SWA | None | **~0.005** |
| **Late QAT** (scale<0.15) | Int6 in warmdown only | No QAT at all | **~0.005** |
| **More TTT epochs** | 10 epochs | 3 epochs | ~0.003 |
| **Partial RoPE** (16 dims) | 16/64 head dims | Full RoPE | ~0.002 |
| **LN Scale** | 1/sqrt(layer+1) | None | ~0.001 |
| **XSA4** | Cross-sequence attn | None | ~0.002 |

### Priority Actions (Ordered by Expected Impact)

1. **Switch TTT from SGD to AdamW** — PR #442 got **0.019 BPB** from this single change alone (SGD→AdamW, lr=0.0005, 10 epochs, no momentum). This is the single biggest gain available.
2. **Add EMA (decay=0.997)** — Exponential moving average of weights, orthogonal to SWA. All top entries use it. Stack: EMA generates smooth weights → SWA averages EMA snapshots.
3. **Add Late QAT (scale<0.15)** — Enable int6 fake-quantization only in the final warmdown phase. Compatible with EMA+SWA stacking. Reduces quant gap from ~0.04 to ~0.007.
4. **Get v24 running on Modal** — Already has Partial RoPE, LN Scale, XSA4. Set `USE_FA3=0` env var to bypass flash-attn compile issues.
5. **Speed optimization** — Profile step time. Our 116ms/step vs SOTA's 82ms means 40% fewer training steps. Check for torch.compile graph breaks with `TORCH_LOGS=graph_breaks`.

---

## File Reference

| File | Description |
|------|-------------|
| `train_gpt_v21.py` | **Best script** (1.1393 BPB). 11L, no-QAT, SWA, TTT, all features |
| `train_gpt_v24.py` | SOTA-based from PR #315/#338. Untested. Needs FA3 fix |
| `train_gpt_v6.py` | Best 10L with QAT (1.1598 BPB) |
| `train_gpt_v14.py` | Best 10L with WD=0.02 (1.1577 BPB) |
| `run_modal.py` | Modal cloud runner for 8×H100 |
| `train_gpt.py` | Upstream baseline (do not modify directly) |

---

## Submission Process

1. **Train on Modal** (or local 8×H100):
   ```bash
   modal run run_modal.py --script train_gpt_v21.py
   ```

2. **Run 3 seeds** for official submission:
   ```bash
   modal run run_modal.py --script train_gpt_v21.py --seed 1337
   modal run run_modal.py --script train_gpt_v21.py --seed 42
   modal run run_modal.py --script train_gpt_v21.py --seed 3
   ```

3. **Create submission record**:
   ```bash
   mkdir -p records/track_10min_16mb/YYYY-MM-DD_Description/
   # Copy: train_gpt.py, README.md (with results table)
   ```

4. **Submit PR** to `openai/parameter-golf`:
   ```bash
   git checkout -b submission/your-branch
   git add records/track_10min_16mb/...
   git push origin submission/your-branch
   # Open PR on GitHub
   ```

5. **Important**: Create a NEW PR for each major update (don't keep editing the same PR).

---

## Git Branches

| Branch | Status |
|--------|--------|
| `submission/seq2048-fp16emb` | PR #63 (1.1598, old) |
| `submission/v12-next` | Current working branch (1.1393 best) |
| `main` | Tracks upstream |

---

## Quick Start on a New Machine

```bash
# 1. Clone
git clone https://github.com/YOUR_FORK/parameter-golf.git
cd parameter-golf
git checkout submission/v12-next

# 2. Install
pip install torch numpy sentencepiece zstandard tqdm modal
pip install flash-attn --no-build-isolation  # optional

# 3. Download data
python data/download.py

# 4. Setup Modal volume (one-time)
modal setup
modal volume create pg-data
modal volume put pg-data ./data/datasets/fineweb10B_sp1024/ /datasets/fineweb10B_sp1024/
modal volume put pg-data ./data/tokenizers/ /tokenizers/

# 5. Run best config
modal run run_modal.py --script train_gpt_v21.py

# 6. Run with overrides
modal run run_modal.py --script train_gpt_v21.py --env "NUM_LAYERS=11,MUON_WD=0.04"
```

---

## Score History

| Date | Version | Config | BPB | Notes |
|------|---------|--------|-----|-------|
| 03-17 | v1 | Baseline 9L | ~1.25 | Starting point |
| 03-18 | v3 | Seq2048 + FP16 embed | ~1.20 | +seq_len, +fp16 |
| 03-19 | v6 | 10L Int6 QAT | 1.1598 | PR #63 submission |
| 03-19 | v8 | 10L Int6 QAT + BigramHash | 1.1577 | WD=0.02 |
| 03-20 | v12 | 11L no-QAT + SWA | ~1.148 | Key breakthrough |
| 03-20 | v17 | 11L + FA3 | 1.1454 | Flash Attention |
| 03-20 | v19 | 11L + SWA tuned | 1.1414 | SWA every 50 steps |
| 03-20 | v21 | 11L + SWA + TTT(freeze=0) | **1.1393** | Current best |
| — | v24 | SOTA stack (PR #338) | untested | Next to try |

---

## Competitive Landscape (2026-03-22)

The competition has evolved significantly. Key patterns from top entries:

### Winning Stack (PR #442, 1.1027 BPB)
Built on PR #398 with a single change: **SGD → AdamW for TTT**. This dropped mean BPB from 1.1221 to 1.1027 — a 0.019 improvement from changing the TTT optimizer alone. Key diff:
```python
# Before (SGD TTT):
optimizer = torch.optim.SGD(ttt_params, lr=0.008, momentum=0.9)
ttt_epochs = 20

# After (AdamW TTT):
optimizer = torch.optim.AdamW(ttt_params, lr=0.0005, weight_decay=0.0)
ttt_epochs = 10
```

### Common Architecture in Top 10
All top entries share: 11L, 512 dim, 8/4 GQA heads, MLP 3x, ReLU², BigramHash 2048, SmearGate, WD=0.04, zstd-22, sliding window stride=64. Differentiation comes from:
- **TTT optimizer choice** (AdamW >> SGD)
- **EMA + SWA stacking** (EMA provides smooth weights, SWA averages EMA snapshots)
- **Late QAT** (fake-quant only in warmdown, reduces quant gap)
- **Partial RoPE** (16/64 dims only)

### Entries to Watch
- **PR #442** (1.1027): Current SOTA, simple AdamW TTT change
- **PR #401** (1.1243): Full architectural stack (EMA + SWA + Late QAT + VE128 + Partial RoPE)
- **PR #445** (1.1232): TTT Burst (replay training batches before eval)
- **PR #450** (1.1466): 12L + Catalytic Residuals (novel architecture)
