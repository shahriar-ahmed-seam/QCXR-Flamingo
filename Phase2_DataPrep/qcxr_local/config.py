"""
config.py  ─  LOCAL (CPU) configuration
Model: DistilGPT-2 (82 M params, ~330 MB)  ← no access required, runs on CPU
Encoder: Swin-Tiny (28 M params)            ← smallest Swin variant
Strategy: pre-compute & cache all image features once upfront → huge CPU speedup
"""
from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT      = Path(r"C:\Users\Seam\Desktop\Research\NSU\data")
IMAGE_DIR      = DATA_ROOT / "images" / "images_normalized"
ANN_PATH       = DATA_ROOT / "annotation.json"
FEATURES_CACHE = DATA_ROOT / "swin_tiny_features_cache.pt"   # pre-computed cache
RESULTS_DIR    = Path("results")

# ── Model selection ───────────────────────────────────────────────────────────
LLM_NAME     = "distilgpt2"                               # 82 M – freely available
ENCODER_NAME = "microsoft/swin-tiny-patch4-window7-224"   # 28 M – smallest Swin

VISUAL_DIM   = 768   # Swin-Tiny last-hidden-state dim
LLM_DIM      = 768   # DistilGPT-2 embedding dim (must match VISUAL_DIM here)

# ── Training ──────────────────────────────────────────────────────────────────
DEVICE            = "cpu"
BATCH_SIZE        = 2          # keep RAM usage low
EPOCHS            = 5          # smoke-test / feasibility run
LR                = 1e-4
MAX_TEXT_LEN      = 60         # matches R2Gen paper default
BEAM_SIZE         = 3
PRECOMPUTE_FEATS  = True       # critical CPU optimisation

# ── Baselines to train (skip VQC today) ──────────────────────────────────────
BOTTLENECKS = ["linear", "mlp", "transformer"]

# ── Transformer bottleneck settings ──────────────────────────────────────────
TRANS_NHEAD   = 8
TRANS_LAYERS  = 2

RESULTS_DIR.mkdir(exist_ok=True)
