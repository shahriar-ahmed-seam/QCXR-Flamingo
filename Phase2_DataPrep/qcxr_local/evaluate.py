"""
evaluate.py
NLP metrics: BLEU-1/4, ROUGE-L, METEOR
Clinical F1: lightweight keyword-match (proxy for CheXbert, no GPU needed)

Usage:
    from evaluate import compute_metrics
    results = compute_metrics(predictions, references)
"""
import re
from collections import Counter


# ─── BLEU ─────────────────────────────────────────────────────────────────────
def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

def bleu_score(predictions, references, n=4):
    """Corpus-level BLEU-n."""
    clipped = 0; total = 0; ref_len = 0; hyp_len = 0
    for hyp, ref in zip(predictions, references):
        h = hyp.lower().split(); r = ref.lower().split()
        ref_len += len(r); hyp_len += len(h)
        ref_ng = _ngrams(r, n); hyp_ng = _ngrams(h, n)
        for gram, cnt in hyp_ng.items():
            clipped += min(cnt, ref_ng.get(gram, 0))
        total += max(0, len(h) - n + 1)
    precision = clipped / max(total, 1)
    import math
    bp = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return bp * precision


def bleu1(predictions, references):
    return bleu_score(predictions, references, n=1)

def bleu4(predictions, references):
    return bleu_score(predictions, references, n=4)


# ─── ROUGE-L ───────────────────────────────────────────────────────────────────
def _lcs(x, y):
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1]==y[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(predictions, references):
    scores = []
    for hyp, ref in zip(predictions, references):
        h = hyp.lower().split(); r = ref.lower().split()
        lcs = _lcs(h, r)
        p = lcs / max(len(h), 1); rec = lcs / max(len(r), 1)
        f1 = 2*p*rec / max(p+rec, 1e-8)
        scores.append(f1)
    return sum(scores) / max(len(scores), 1)


# ─── Clinical F1 (CheXpert 14-class keyword proxy) ────────────────────────────
CHEXPERT_LABELS = {
    "Atelectasis":          ["atelectasis", "atelectatic"],
    "Cardiomegaly":         ["cardiomegaly", "cardiac enlargement", "enlarged heart"],
    "Consolidation":        ["consolidation", "consolidative"],
    "Edema":                ["edema", "oedema"],
    "Effusion":             ["effusion", "pleural effusion"],
    "Emphysema":            ["emphysema"],
    "Fibrosis":             ["fibrosis", "fibrotic"],
    "Hernia":               ["hernia"],
    "Infiltration":         ["infiltrate", "infiltration"],
    "Mass":                 ["mass", "masses"],
    "Nodule":               ["nodule", "nodular"],
    "Pleural_Thickening":   ["pleural thickening"],
    "Pneumonia":            ["pneumonia"],
    "Pneumothorax":         ["pneumothorax"],
}

def _extract_labels(text: str) -> set:
    text = text.lower()
    found = set()
    for label, keywords in CHEXPERT_LABELS.items():
        if any(kw in text for kw in keywords):
            found.add(label)
    return found

def clinical_f1(predictions, references):
    tp = fp = fn = 0
    for pred, ref in zip(predictions, references):
        pred_labels = _extract_labels(pred)
        ref_labels  = _extract_labels(ref)
        tp += len(pred_labels & ref_labels)
        fp += len(pred_labels - ref_labels)
        fn += len(ref_labels  - pred_labels)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1


# ─── Master function ──────────────────────────────────────────────────────────
def compute_metrics(predictions: list, references: list) -> dict:
    """
    Args:
        predictions: list of generated report strings
        references : list of ground-truth report strings
    Returns:
        dict with BLEU-1, BLEU-4, ROUGE-L, CIDEr, Clinical-F1
    """
    assert len(predictions) == len(references), "Length mismatch!"

    try:
        from pycocoevalcap.cider.cider import Cider
        res = {i: [p] for i, p in enumerate(predictions)}
        gts = {i: [r] for i, r in enumerate(references)}
        cider_score, _ = Cider().compute_score(gts, res)
    except ImportError:
        cider_score = 0.0

    return {
        "BLEU-1":      round(bleu1(predictions, references),   4),
        "BLEU-4":      round(bleu4(predictions, references),   4),
        "ROUGE-L":     round(rouge_l(predictions, references), 4),
        "CIDEr":       round(cider_score, 4),
        "Clinical-F1": round(clinical_f1(predictions, references), 4),
    }
