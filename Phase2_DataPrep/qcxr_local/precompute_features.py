"""
precompute_features.py  ─  CPU KEY OPTIMISATION
Runs through ALL images once, passes through frozen Swin-Tiny,
saves a dict {uid → tensor[N, D]} to disk.

After this script runs once (~25 min on CPU), every training epoch is
vastly faster because it only loads cached tensors instead of re-running Swin.

For IU-Xray: 3376 studies × 2 images each → 6752 forward passes.
Expected time: ~30 minutes on Core i5. After that, negligible per epoch.

Usage:
    python precompute_features.py
"""
import json
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import SwinModel
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg

# ── Load frozen encoder ────────────────────────────────────────────────────────
print(f"Loading encoder: {cfg.ENCODER_NAME} ...")
encoder = SwinModel.from_pretrained(cfg.ENCODER_NAME)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ── Load annotation ───────────────────────────────────────────────────────────
ann = json.loads(cfg.ANN_PATH.read_text(encoding="utf-8"))
all_examples = ann["train"] + ann["val"] + ann["test"]
print(f"Total studies to process: {len(all_examples)}")

# ── Pre-compute ───────────────────────────────────────────────────────────────
cache = {}   # uid → mean-pooled feature tensor [N, D]

with torch.no_grad():
    for ex in tqdm(all_examples, desc="Pre-computing Swin features"):
        uid = ex["id"]
        feats = []
        for fname in ex["image_path"]:
            img_path = cfg.IMAGE_DIR / fname
            img = Image.open(img_path).convert("RGB")
            pixel_values = transform(img).unsqueeze(0)   # [1, 3, 224, 224]
            out = encoder(pixel_values=pixel_values)
            feat = out.last_hidden_state.squeeze(0)      # [N, D]
            feats.append(feat)
        # Average frontal + lateral features → single [N, D] tensor per study
        combined = torch.stack(feats).mean(0)            # [N, D]
        cache[uid] = combined.half()                     # save as fp16 → smaller file

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(cache, cfg.FEATURES_CACHE)
print(f"\nSaved {len(cache)} feature tensors to: {cfg.FEATURES_CACHE}")
size_mb = cfg.FEATURES_CACHE.stat().st_size / 1024 / 1024
print(f"Cache file size: {size_mb:.1f} MB")
print("Done. Run train.py next.")
