"""
build_annotation.py
--------------------
Converts the Indiana University CXR Kaggle CSV files into the
annotation.json format required by R2Gen / R2GenGPT / QCXR-Flamingo.

Input files (already in data/):
  - indiana_projections.csv  → uid, filename, projection
  - indiana_reports.csv      → uid, findings, impression, ...

Output:
  - annotation.json          → {train: [...], val: [...], test: [...]}

R2Gen annotation schema (from modules/datasets.py):
  Each record must have:
    "id"         : str  – unique study identifier
    "image_path" : list – [frontal_filename, lateral_filename]
    "report"     : str  – combined findings + impression text

Split: 7:1:2  (train / val / test) on unique UIDs — matches R2Gen paper.
"""

import json
import random
import re
import pandas as pd
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\Seam\Desktop\Research\NSU\data")
IMAGE_DIR  = DATA_DIR / "images" / "images_normalized"
OUT_PATH   = DATA_DIR / "annotation.json"

PROJ_CSV   = DATA_DIR / "indiana_projections.csv"
REP_CSV    = DATA_DIR / "indiana_reports.csv"

SEED = 42
random.seed(SEED)

# ── Load CSVs ────────────────────────────────────────────────────────────────
proj = pd.read_csv(PROJ_CSV)   # uid, filename, projection
rep  = pd.read_csv(REP_CSV)    # uid, findings, impression, ...

print(f"Projections rows : {len(proj)}")
print(f"Reports rows     : {len(rep)}")

# ── Build uid → {frontal, lateral} map ─────────────────────────────────────
uid_images = {}
for _, row in proj.iterrows():
    uid = str(row["uid"])
    fname = row["filename"]
    ptype = str(row["projection"]).strip().lower()   # 'frontal' or 'lateral'
    uid_images.setdefault(uid, {})
    if "frontal" in ptype:
        uid_images[uid]["frontal"] = fname
    elif "lateral" in ptype:
        uid_images[uid]["lateral"] = fname

# ── Text cleaning (mirrors R2Gen tokenizers.py clean_report_iu_xray) ────────
def clean_report(text: str) -> str:
    if not isinstance(text, str) or text.strip().lower() in ("nan", "none", ""):
        return ""
    t = text.replace("..", ".").replace("1. ", "").replace(". 2. ", ". ") \
            .replace(". 3. ", ". ").replace(". 4. ", ". ").replace(". 5. ", ". ") \
            .replace(" 2. ", ". ").replace(" 3. ", ". ").replace(" 4. ", ". ") \
            .replace(" 5. ", ". ").strip()
    return t

# ── Build per-uid report text ────────────────────────────────────────────────
uid_reports = {}
for _, row in rep.iterrows():
    uid = str(row["uid"])
    findings   = clean_report(str(row.get("findings",   "")))
    impression = clean_report(str(row.get("impression", "")))
    # Use findings; fall back to impression; combine when both exist
    if findings and impression:
        report = findings + " " + impression
    elif findings:
        report = findings
    elif impression:
        report = impression
    else:
        report = ""
    uid_reports[uid] = report.strip()

# ── Build valid examples (need BOTH views + non-empty report) ────────────────
examples = []
missing_lateral = 0
missing_frontal = 0
empty_report    = 0
missing_image   = 0

for uid, imgs in uid_images.items():
    frontal = imgs.get("frontal")
    lateral = imgs.get("lateral")

    if not frontal:
        missing_frontal += 1
        continue
    if not lateral:
        missing_lateral += 1
        continue

    report = uid_reports.get(uid, "")
    if not report:
        empty_report += 1
        continue

    # Verify image files actually exist on disk
    if not (IMAGE_DIR / frontal).exists() or not (IMAGE_DIR / lateral).exists():
        missing_image += 1
        continue

    examples.append({
        "id"         : uid,
        "image_path" : [frontal, lateral],   # frontal first, lateral second
        "report"     : report
    })

print(f"\nValid examples   : {len(examples)}")
print(f"  Missing frontal: {missing_frontal}")
print(f"  Missing lateral: {missing_lateral}")
print(f"  Empty report   : {empty_report}")
print(f"  Image not found: {missing_image}")

# ── 7 : 1 : 2 split ──────────────────────────────────────────────────────────
random.shuffle(examples)
n     = len(examples)
n_test = int(n * 0.2)
n_val  = int(n * 0.1)
n_train = n - n_test - n_val

test  = examples[:n_test]
val   = examples[n_test : n_test + n_val]
train = examples[n_test + n_val:]

print(f"\nSplit  → train: {len(train)}  val: {len(val)}  test: {len(test)}")

# ── Dump JSON ────────────────────────────────────────────────────────────────
annotation = {"train": train, "val": val, "test": test}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(annotation, f, indent=2, ensure_ascii=False)

print(f"\nSaved: {OUT_PATH}")
print("\nSample train record:")
print(json.dumps(train[0], indent=2))
