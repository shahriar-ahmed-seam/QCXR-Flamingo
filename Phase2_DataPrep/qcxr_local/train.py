"""
train.py  ─  LOCAL (CPU) training script
Trains one bottleneck type for EPOCHS epochs.

Usage:
    python train.py --bottleneck linear
    python train.py --bottleneck mlp
    python train.py --bottleneck transformer
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from dataset import IUXrayDataset, collate_fn
from models.qcxr_model import QCXRModel
from models.encoder import get_transforms
from evaluate import compute_metrics


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bottleneck", type=str, default="linear",
                   choices=cfg.BOTTLENECKS)
    p.add_argument("--epochs",     type=int, default=cfg.EPOCHS)
    p.add_argument("--batch_size", type=int, default=cfg.BATCH_SIZE)
    p.add_argument("--lr",         type=float, default=cfg.LR)
    return p.parse_args()


def load_features_cache():
    """Load pre-computed Swin features if available."""
    if cfg.PRECOMPUTE_FEATS and cfg.FEATURES_CACHE.exists():
        print(f"Loading cached features from {cfg.FEATURES_CACHE} ...")
        cache = torch.load(cfg.FEATURES_CACHE, map_location="cpu")
        # Convert fp16 → fp32 for CPU stability
        cache = {k: v.float() for k, v in cache.items()}
        return cache
    elif cfg.PRECOMPUTE_FEATS:
        print("WARNING: PRECOMPUTE_FEATS=True but cache file missing!")
        print("Run:  python precompute_features.py  first.")
        return None
    return None


def build_loader(split, tokenizer, cache, batch_size, shuffle):
    ds = IUXrayDataset(
        ann_path=cfg.ANN_PATH,
        image_dir=cfg.IMAGE_DIR,
        tokenizer=tokenizer,
        split=split,
        transform=get_transforms(split) if cache is None else None,
        features_cache=cache,
        max_text_len=cfg.MAX_TEXT_LEN,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_fn, num_workers=0, pin_memory=False)


def train_epoch(model, loader, optimizer, device, bottleneck_type, encoder=None):
    """One training epoch. Returns mean loss."""
    model.bottleneck.train()
    total_loss = 0.0

    for batch_idx, (uids, visuals, input_ids, attn_mask, labels) in enumerate(loader):
        # If live images (no cache), run encoder
        if visuals.dim() == 5:   # [B, 2, C, H, W]
            B, V, C, H, W = visuals.shape
            flat = visuals.view(B * V, C, H, W).to(device)
            with torch.no_grad():
                feats = encoder(flat)              # [B*V, N, D]
            B2, N, D = feats.shape
            feats = feats.view(B, V, N, D).mean(1) # [B, N, D] mean frontal+lateral
        else:
            feats = visuals.to(device)             # [B, N, D] cached

        input_ids  = input_ids.to(device)
        attn_mask  = attn_mask.to(device)
        labels     = labels.to(device)

        optimizer.zero_grad()
        loss = model(feats, input_ids, attn_mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 50 == 0:
            print(f"  step {batch_idx+1}/{len(loader)}  loss={loss.item():.4f}")

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, encoder=None):
    """Run generation on val/test split and compute metrics."""
    model.eval()
    preds, refs = [], []

    for uids, visuals, input_ids, attn_mask, labels in loader:
        if visuals.dim() == 5:
            B, V, C, H, W = visuals.shape
            flat = visuals.view(B*V, C, H, W).to(device)
            feats = encoder(flat)
            B2, N, D = feats.shape
            feats = feats.view(B, V, N, D).mean(1)
        else:
            feats = visuals.to(device)

        generated = model.generate(feats, max_new_tokens=cfg.MAX_TEXT_LEN,
                                   beam_size=cfg.BEAM_SIZE)
        preds.extend(generated)

        # Decode reference from labels (ignore -100)
        for lbl in labels:
            tok = lbl[lbl != -100].tolist()
            refs.append(model.tokenizer.decode(tok, skip_special_tokens=True))

    return compute_metrics(preds, refs)


def run(args):
    device = torch.device(cfg.DEVICE)
    print(f"\n{'='*55}")
    print(f"  QCXR-Local  |  bottleneck={args.bottleneck.upper()}  |  device={cfg.DEVICE}")
    print(f"{'='*55}")

    # ── Load feature cache ────────────────────────────────────────────────────
    cache = load_features_cache()

    # ── Build model ───────────────────────────────────────────────────────────
    print(f"Loading LLM: {cfg.LLM_NAME} ...")
    model = QCXRModel(
        llm_name=cfg.LLM_NAME,
        bottleneck_name=args.bottleneck,
        visual_dim=cfg.VISUAL_DIM,
        nhead=cfg.TRANS_NHEAD,
        trans_layers=cfg.TRANS_LAYERS,
    ).to(device)
    model.tokenizer.pad_token = model.tokenizer.eos_token

    print(f"Trainable params: {model.trainable_params():,}")

    # Live encoder only needed if no cache
    encoder = None
    if cache is None:
        from models.encoder import FrozenSwinEncoder
        encoder = FrozenSwinEncoder(cfg.ENCODER_NAME).to(device)

    # ── Data loaders ──────────────────────────────────────────────────────────
    train_loader = build_loader("train", model.tokenizer, cache,
                                args.batch_size, shuffle=True)
    val_loader   = build_loader("val",   model.tokenizer, cache,
                                args.batch_size, shuffle=False)

    # ── Optimiser (bottleneck params only) ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_bleu4 = 0.0
    results_per_epoch = []
    ckpt_path = cfg.RESULTS_DIR / f"best_{args.bottleneck}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device,
                                 args.bottleneck, encoder)
        scheduler.step()
        elapsed = time.time() - t0

        val_metrics = evaluate(model, val_loader, device, encoder)
        print(f"  train_loss={train_loss:.4f}  val={val_metrics}  ({elapsed:.0f}s)")

        results_per_epoch.append({"epoch": epoch, "loss": train_loss, **val_metrics})

        if val_metrics["BLEU-4"] > best_bleu4:
            best_bleu4 = val_metrics["BLEU-4"]
            torch.save({"bottleneck_state": model.bottleneck.state_dict(),
                        "metrics": val_metrics}, ckpt_path)
            print(f"  *** New best BLEU-4={best_bleu4:.4f} — checkpoint saved ***")

    # ── Final eval on test set ────────────────────────────────────────────────
    print("\nLoading best checkpoint for test evaluation ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.bottleneck.load_state_dict(ckpt["bottleneck_state"])
    test_loader = build_loader("test", model.tokenizer, cache,
                               args.batch_size, shuffle=False)
    test_metrics = evaluate(model, test_loader, device, encoder)
    print(f"TEST metrics: {test_metrics}")

    # ── Save to CSV ───────────────────────────────────────────────────────────
    csv_path = cfg.RESULTS_DIR / "metrics_table.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "BLEU-1", "BLEU-4",
                                               "ROUGE-L", "Clinical-F1"])
        if write_header:
            writer.writeheader()
        writer.writerow({"Model": args.bottleneck, **test_metrics})

    print(f"\nResults appended to {csv_path}")
    return test_metrics


if __name__ == "__main__":
    args = get_args()
    run(args)
