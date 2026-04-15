"""
dataset.py  ─  IU X-Ray PyTorch Dataset
Loads annotation.json and serves (image_features OR images, report) pairs.
Supports two modes:
  1. LIVE mode   : loads images on-the-fly (used on GPU / Kaggle)
  2. CACHED mode : returns pre-computed Swin features (critical for CPU)
"""
import json
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class IUXrayDataset(Dataset):
    """
    Args:
        ann_path    : path to annotation.json
        image_dir   : directory containing PNG images
        tokenizer   : HuggingFace tokenizer for the LLM
        split       : 'train' | 'val' | 'test'
        transform   : torchvision transform for raw images
        features_cache : dict {uid: tensor [N, D]} from precompute_features.py
                        If provided, images are NOT loaded from disk.
        max_text_len : max tokens for the report
    """

    def __init__(self, ann_path, image_dir, tokenizer, split,
                 transform=None, features_cache=None, max_text_len=60):
        self.image_dir      = Path(image_dir)
        self.tokenizer      = tokenizer
        self.split          = split
        self.transform      = transform
        self.features_cache = features_cache   # dict or None
        self.max_text_len   = max_text_len

        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        self.examples = ann[split]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex     = self.examples[idx]
        uid    = ex["id"]
        report = ex["report"]

        # ── Visual features ───────────────────────────────────────────────────
        if self.features_cache is not None:
            # CACHED mode: tensor [N, D] ready to use
            visual = self.features_cache[uid]           # [N, D]
        else:
            # LIVE mode: load both frontal + lateral, stack → [2, C, H, W]
            imgs = []
            for fname in ex["image_path"]:
                img = Image.open(self.image_dir / fname).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                imgs.append(img)
            visual = torch.stack(imgs, dim=0)           # [2, C, H, W]

        # ── Tokenise report ───────────────────────────────────────────────────
        enc = self.tokenizer(
            report,
            max_length=self.max_text_len,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)          # [T]
        # labels = input_ids (causal LM: predict every token)
        # mask = -100 on padding
        labels = input_ids.clone()

        return uid, visual, input_ids, labels


def collate_fn(batch):
    """Pads reports to the same length within a batch."""
    uids, visuals, input_ids_list, labels_list = zip(*batch)

    # Pad text
    pad_id = 0   # will be overridden to tokenizer.pad_token_id in train.py
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    labels    = pad_sequence(labels_list,    batch_first=True, padding_value=-100)
    attn_mask = (input_ids != pad_id).long()

    # Visual: either [B, N, D] (cached) or [B, 2, C, H, W] (live)
    visuals = torch.stack(visuals, dim=0)

    return uids, visuals, input_ids, attn_mask, labels
