"""
run_all_baselines.py  ─  Runs all 3 baselines sequentially and prints final table
Usage:  python run_all_baselines.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from train import run
import argparse
import csv

print("="*60)
print("  QCXR-Flamingo | Run All 3 Classical Baselines (LOCAL CPU)")
print("="*60)

all_results = {}
for bt in cfg.BOTTLENECKS:
    print(f"\n\n{'#'*60}")
    print(f"#  TRAINING: {bt.upper()} BOTTLENECK")
    print(f"{'#'*60}")

    class FakeArgs:
        bottleneck = bt
        epochs     = cfg.EPOCHS
        batch_size = cfg.BATCH_SIZE
        lr         = cfg.LR

    metrics = run(FakeArgs())
    all_results[bt] = metrics

# ── Pretty table ──────────────────────────────────────────────────────────────
print("\n\n" + "="*60)
print("  FINAL RESULTS TABLE")
print("="*60)
print(f"{'Model':<20} {'BLEU':>8} {'ROUGE':>8} {'CIDEr':>8} {'Clin-F1':>8}")
print("-" * 65)
for name, m in all_results.items():
    print(f"{name:<20} {m['BLEU-4']:>8.4f} {m['ROUGE-L']:>8.4f} "
          f"{m['CIDEr']:>8.4f} {m['Clinical-F1']:>8.4f}")
print("=" * 65)
print(f"\nDetailed CSV saved to: {cfg.RESULTS_DIR}/metrics_table.csv")
