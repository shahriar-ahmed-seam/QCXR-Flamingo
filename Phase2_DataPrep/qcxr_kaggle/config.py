"""
config.py  ─  KAGGLE (GPU) configuration
Model: TinyLlama-1.1B-Chat  (1.1 B params, ~2.2 GB fp16)  ← freely available, no approval
Encoder: Swin-Base            (87 M params)                ← matches the paper
GPU target: Kaggle T4 / P100 (15–16 GB VRAM)
"""
from pathlib import Path

# ── Paths (Kaggle online environment) ─────────────────────────────────────────
# Upload annotation.json to Kaggle as a Dataset named "qcxr-annotation"
DATA_ROOT = Path("/kaggle/input/chest-xrays-indiana-university")
IMAGE_DIR = DATA_ROOT / "images" / "images_normalized"
ANN_PATH  = Path("/kaggle/input/qcxr-annotation/annotation.json")
RESULTS_DIR = Path("/kaggle/working/results")

# ── Model selection ───────────────────────────────────────────────────────────
LLM_NAME     = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"       # freely available
ENCODER_NAME = "microsoft/swin-base-patch4-window7-224"    # paper-faithful

VISUAL_DIM = 1024   # Swin-Base last-hidden-state dim
LLM_DIM    = 2048   # TinyLlama embedding dim

# ── Training ──────────────────────────────────────────────────────────────────
DEVICE           = "cuda"
BATCH_SIZE       = 4
EPOCHS           = 15          # full training as per paper
LR               = 1e-4
MAX_TEXT_LEN     = 60
BEAM_SIZE        = 3
PRECOMPUTE_FEATS = False       # GPU is fast – no need

# ── Baselines ─────────────────────────────────────────────────────────────────
BOTTLENECKS = ["linear", "mlp", "transformer"]

# ── Transformer bottleneck settings ──────────────────────────────────────────
TRANS_NHEAD  = 8
TRANS_LAYERS = 2

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
