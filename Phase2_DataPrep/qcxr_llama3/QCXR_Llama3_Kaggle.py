"""
QCXR_Llama3_Kaggle.py
══════════════════════════════════════════════════════════════════════════════
Paste this ENTIRE file into a single Kaggle code cell.
LLM backbone: meta-llama/Llama-3.1-8B  (4-bit quantized via bitsandbytes)
Vision encoder: microsoft/swin-base-patch4-window7-224  (frozen)
Bottlenecks trained: linear, mlp, transformer, vqc (QCXR-Flamingo)

⚠️  KAGGLE SETUP REQUIREMENTS (read the Setup Guide below the code):
    1. Enable GPU T4 x2 (or single T4 with 4-bit quant)
    2. Add HuggingFace token as a Kaggle Secret named HF_TOKEN
    3. Attach datasets: chest-xrays-indiana-university + qcxr-annotation
══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1 ─ Install dependencies
# ═══════════════════════════════════════════════════════════════════════════════
import subprocess
subprocess.run([
    "pip", "install", "-q",
    "transformers>=4.40.0",   # Llama-3.1 support added in 4.40
    "accelerate",
    "bitsandbytes",           # 4-bit quantization (critical for 8B on T4)
    "pycocoevalcap",          # CIDEr metric
    "tqdm",
    "pennylane",              # Quantum circuit simulation (QCXR-Flamingo VQC)
], check=False)

import torch
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ONLY ⚠️")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2 ─ Authenticate with HuggingFace (Llama-3.1-8B is a gated model)
# ═══════════════════════════════════════════════════════════════════════════════
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

hf_token = UserSecretsClient().get_secret("HF_TOKEN")
login(token=hf_token, add_to_git_credential=False)
print("✓ Logged into HuggingFace as the model owner.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3 ─ Config
# ═══════════════════════════════════════════════════════════════════════════════
import os
from pathlib import Path

DATA_ROOT = Path("/kaggle/input/datasets/raddar/chest-xrays-indiana-university")

# ── Auto-detect image folder ──────────────────────────────────────────────────
IMAGE_DIR = None
for root, dirs, files in os.walk(DATA_ROOT):
    png_files = [f for f in files if f.endswith(".png")]
    if len(png_files) > 10:
        IMAGE_DIR = Path(root)
        print(f"✓ Images found ({len(png_files)} PNGs): {IMAGE_DIR}")
        break
if IMAGE_DIR is None:
    raise FileNotFoundError("PNG images not found. Check dataset attachment.")

# ── annotation.json ───────────────────────────────────────────────────────────
_candidates = [
    Path("/kaggle/input/qcxr-annotation/annotation.json"),
    Path("/kaggle/input/qcxr-annotation/data/annotation.json"),
    Path("/kaggle/working/annotation.json"),
]
ANN_PATH = next((p for p in _candidates if p.exists()), None)
if ANN_PATH is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "annotation.json" in files:
            ANN_PATH = Path(root) / "annotation.json"
            break
if ANN_PATH is None:
    raise FileNotFoundError("annotation.json not found. Attach qcxr-annotation dataset.")
print(f"✓ annotation.json: {ANN_PATH}")

RESULTS = Path("/kaggle/working/results_llama3")
RESULTS.mkdir(exist_ok=True)

# ── Model choices ─────────────────────────────────────────────────────────────
LLM_NAME      = "meta-llama/Llama-3.1-8B"
ENCODER_NAME  = "microsoft/swin-base-patch4-window7-224"
VISUAL_DIM    = 1024       # Swin-Base hidden dim
LLM_DIM       = 4096       # Llama-3.1-8B hidden dim

# ── Training hyperparameters ──────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ── Select which bottleneck to train in this session ───────────
# Options: "linear", "mlp", "transformer", "vqc"
# NOTE: "vqc" = QCXR-Flamingo quantum bottleneck (slower, ~2x epoch time)
CURRENT_BOTTLENECK = "linear" 

BATCH_SIZE    = 4          # Slightly increased for T4 x2
EPOCHS        = 12         # Reduced from 15 for Kaggle 12h safety margin
LR            = 5e-5       
MAX_TEXT_LEN  = 60
BEAM_SIZE     = 1          # Set to 1 for 3x faster training; use 3 only for final test

print(f"Device: {DEVICE} | Batch: {BATCH_SIZE} | LLM: {LLM_NAME}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4 ─ Frozen Swin Encoder
# ═══════════════════════════════════════════════════════════════════════════════
import torch.nn as nn
from transformers import SwinModel

class FrozenSwinEncoder(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.model = SwinModel.from_pretrained(name)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        self.hidden_dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state

print("FrozenSwinEncoder defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5 ─ Bottleneck modules
# ═══════════════════════════════════════════════════════════════════════════════
class LinearBottleneck(nn.Module):
    def __init__(self, vd, ld):
        super().__init__()
        self.proj = nn.Linear(vd, ld)

    def forward(self, x):
        return self.proj(x.mean(1)).unsqueeze(1)


class MLPBottleneck(nn.Module):
    def __init__(self, vd, ld, hd=1024):
        super().__init__()
        # Larger hidden dim to match Llama's 4096 space
        self.mlp = nn.Sequential(
            nn.Linear(vd, hd), nn.GELU(), nn.LayerNorm(hd),
            nn.Linear(hd, hd), nn.GELU(), nn.LayerNorm(hd),
            nn.Linear(hd, ld),
        )

    def forward(self, x):
        return self.mlp(x.mean(1)).unsqueeze(1)


class TransformerBottleneck(nn.Module):
    def __init__(self, vd, ld, nh=8, nl=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=vd, nhead=nh, dim_feedforward=vd * 2,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.tf   = nn.TransformerEncoder(layer, num_layers=nl)
        self.proj = nn.Linear(vd, ld)

    def forward(self, x):
        return self.proj(self.tf(x).mean(1)).unsqueeze(1)


# ── QCXR-Flamingo VQC Bottleneck ─────────────────────────────────────────────
# Implements the quantum pipeline from the QCXR-Flamingo paper:
#   Image → Encoder → Reducer → VQC → Projection → LM
#
# Pipeline (per paper notation):
#   zi = W·fi              (Feature Compression: vd → n_qubits)
#   |ψ(x)⟩ = ⊗ Ry(zj)|0⟩  (Quantum Encoding via Ry rotation gates)
#   |ψθ⟩   = U(θ)|ψ(x)⟩   (Variational Circuit with trainable θ)
#   qi     = ⟨ψθ|Zi|ψθ⟩   (Measurement: PauliZ expectation values)
#   hv     = Wq·q          (Projection: n_qubits → LLM_DIM)
# ─────────────────────────────────────────────────────────────────────────────
import pennylane as qml
import math

class VQCBottleneck(nn.Module):
    def __init__(self, vd, ld, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers

        # Classical reducer: vd → n_qubits (zi = W·fi)
        self.reducer   = nn.Linear(vd, n_qubits)

        # Variational parameters θ: shape [n_layers, n_qubits, 3] (Rot gate: Rz Ry Rz)
        self.weights   = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.01
        )

        # Classical projection: n_qubits → ld (hv = Wq·q)
        self.proj      = nn.Linear(n_qubits, ld)

        # PennyLane simulator device
        self.dev       = qml.device("default.qubit", wires=n_qubits)

        # Build and JIT-compile the QNode
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def _circuit(inputs, weights):
            # ── Encoding layer: Ry(zi)|0⟩ ────────────────────────────
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # ── Variational layers: U(θ) ──────────────────────────────
            for layer in range(n_layers):
                # Trainable Rot gates (Rz·Ry·Rz decomposition)
                for i in range(n_qubits):
                    qml.Rot(weights[layer, i, 0],
                            weights[layer, i, 1],
                            weights[layer, i, 2], wires=i)
                # Entanglement: linear CNOT chain
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

            # ── Measurement: PauliZ expectation values qi = ⟨ψθ|Zi|ψθ⟩
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = _circuit

    def forward(self, x):
        # x: [B, N, vd]  →  pool  →  [B, vd]
        x   = x.mean(1).float()

        # Feature compression + scale to [-π, π] for stable Ry encoding
        z   = torch.tanh(self.reducer(x)) * math.pi  # [B, n_qubits]

        # Run quantum circuit sample-by-sample (PennyLane needs 1-D inputs)
        q_out = torch.stack([
            torch.stack(self._circuit(z[b], self.weights))
            for b in range(z.size(0))
        ])  # [B, n_qubits]

        # Project quantum output to LLM embedding dimension
        # .float() is required: PennyLane returns float64 (Double), but
        # self.proj weights are float32 — dtype mismatch causes RuntimeError
        return self.proj(q_out.float()).unsqueeze(1)  # [B, 1, ld]


def get_bottleneck(name, vd, ld):
    return {
        "linear":      LinearBottleneck(vd, ld),
        "mlp":         MLPBottleneck(vd, ld),
        "transformer": TransformerBottleneck(vd, ld),
        "vqc":         VQCBottleneck(vd, ld, n_qubits=4, n_layers=2),
    }[name]

print("Bottleneck classes defined (Linear | MLP | Transformer | VQC).")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6 ─ QCXR Model with Llama-3.1-8B (4-bit quantized)
# ═══════════════════════════════════════════════════════════════════════════════
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)

# 4-bit quantization config → reduces 8B model from ~16GB to ~5GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLM weights
    bnb_4bit_compute_dtype=torch.float16, # Computations still in fp16
    bnb_4bit_use_double_quant=True,       # Nested quantization — saves extra ~0.4GB
)

class QCXRLlama3Model(nn.Module):
    def __init__(self, llm_name, bottleneck_name):
        super().__init__()

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            # Llama-3 uses <|end_of_text|> as EOS; set pad = EOS
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── LLM (frozen, 4-bit quantized) ─────────────────────────────────────
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            quantization_config=bnb_config,
            device_map="auto",            # Handles multi-GPU / CPU offload automatically
            torch_dtype=torch.float16,
        )
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()

        # ── Trainable Bottleneck (float32 for gradient stability) ─────────────
        self.bottleneck = get_bottleneck(bottleneck_name, VISUAL_DIM, LLM_DIM).float()

    def _embed(self):
        # Llama-3 embedding layer path (same as Llama-2)
        return self.llm.model.embed_tokens

    def forward(self, vis_feats, input_ids, attn_mask, labels):
        vis_token  = self.bottleneck(vis_feats.float())       # [B, 1, 4096] float32
        text_emb   = self._embed()(input_ids)                  # [B, T, 4096] fp16
        inputs_emb = torch.cat([vis_token.to(text_emb.dtype), text_emb], dim=1)

        B  = input_ids.size(0)
        vm = torch.ones(B, 1, device=input_ids.device, dtype=attn_mask.dtype)
        vl = torch.full((B, 1), -100, device=input_ids.device, dtype=labels.dtype)

        return self.llm(
            inputs_embeds=inputs_emb,
            attention_mask=torch.cat([vm, attn_mask], 1),
            labels=torch.cat([vl, labels], 1),
        ).loss

    @torch.no_grad()
    def generate(self, vis_feats, max_new_tokens=100, beam_size=3):
        B   = vis_feats.size(0)
        vt  = self.bottleneck(vis_feats.float())
        bos = torch.full(
            (B, 1),
            self.tokenizer.bos_token_id or self.tokenizer.convert_tokens_to_ids("<|begin_of_text|>"),
            device=vis_feats.device, dtype=torch.long
        )
        bos_e = self._embed()(bos)
        ie    = torch.cat([vt.to(bos_e.dtype), bos_e], 1)
        am    = torch.ones(B, ie.size(1), device=vis_feats.device, dtype=torch.long)

        gen = self.llm.generate(
            inputs_embeds=ie,
            attention_mask=am,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)

print("QCXRLlama3Model defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7 ─ Dataset
# ═══════════════════════════════════════════════════════════════════════════════
import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class IUXrayDataset(Dataset):
    def __init__(self, split, tokenizer, transform=None):
        ann = json.loads(ANN_PATH.read_text())
        self.examples  = ann[split]
        self.tokenizer = tokenizer
        self.transform = transform
        # Validate first few paths
        for ex in self.examples[:3]:
            for f in ex["image_path"]:
                p = IMAGE_DIR / f
                if not p.exists():
                    raise FileNotFoundError(f"Image not found: {p}\nIMAGE_DIR={IMAGE_DIR}")

    def __len__(self): return len(self.examples)

    def __getitem__(self, i):
        ex   = self.examples[i]
        imgs = []
        for f in ex["image_path"]:
            img = Image.open(IMAGE_DIR / f).convert("RGB")
            if self.transform: img = self.transform(img)
            imgs.append(img)
        visual = torch.stack(imgs)
        enc = self.tokenizer(
            ex["report"], max_length=MAX_TEXT_LEN,
            truncation=True, return_tensors="pt"
        )
        ids = enc["input_ids"].squeeze(0)
        return ex["id"], visual, ids, ids.clone()


def collate(batch):
    ids, vis, inp, lbl = zip(*batch)
    inp  = pad_sequence(inp, batch_first=True, padding_value=0)
    lbl  = pad_sequence(lbl, batch_first=True, padding_value=-100)
    attn = (inp != 0).long()
    return ids, torch.stack(vis), inp, attn, lbl


train_tf = transforms.Compose([
    transforms.Resize(256), transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
val_tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
print("Dataset class defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8 ─ Evaluation metrics
# ═══════════════════════════════════════════════════════════════════════════════
from collections import Counter
import math

def ngrams(tok, n):
    return Counter(tuple(tok[i:i+n]) for i in range(len(tok) - n + 1))

def bleu_n(preds, refs, n):
    cl = to = rl = hl = 0
    for h, r in zip(preds, refs):
        h = h.lower().split(); r = r.lower().split()
        rl += len(r); hl += len(h)
        rng = ngrams(r, n); hng = ngrams(h, n)
        for g, c in hng.items(): cl += min(c, rng.get(g, 0))
        to += max(0, len(h) - n + 1)
    p  = cl / max(to, 1)
    bp = 1.0 if hl >= rl else math.exp(1 - rl / max(hl, 1))
    return bp * p

def lcs(x, y):
    m, n = len(x), len(y)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if x[i-1]==y[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]

def rouge_l(preds, refs):
    sc = []
    for h, r in zip(preds, refs):
        h=h.lower().split(); r=r.lower().split()
        l=lcs(h,r); p=l/max(len(h),1); rc=l/max(len(r),1)
        sc.append(2*p*rc/max(p+rc,1e-8))
    return sum(sc)/max(len(sc),1)

KEYWORDS = {
    "Atelectasis":["atelectasis"],"Cardiomegaly":["cardiomegaly"],
    "Consolidation":["consolidation"],"Edema":["edema"],
    "Effusion":["effusion"],"Emphysema":["emphysema"],
    "Fibrosis":["fibrosis"],"Infiltration":["infiltrate"],
    "Mass":["mass"],"Nodule":["nodule"],
    "Pleural_Thickening":["pleural thickening"],
    "Pneumonia":["pneumonia"],"Pneumothorax":["pneumothorax"],
}

def clin_f1(preds, refs):
    tp=fp=fn=0
    for p,r in zip(preds,refs):
        pl={l for l,kws in KEYWORDS.items() if any(k in p.lower() for k in kws)}
        rl={l for l,kws in KEYWORDS.items() if any(k in r.lower() for k in kws)}
        tp+=len(pl&rl); fp+=len(pl-rl); fn+=len(rl-pl)
    pr=tp/max(tp+fp,1); re=tp/max(tp+fn,1)
    return 2*pr*re/max(pr+re,1e-8)

def compute_metrics(preds, refs):
    try:
        from pycocoevalcap.cider.cider import Cider
        res={i:[p] for i,p in enumerate(preds)}
        gts={i:[r] for i,r in enumerate(refs)}
        cider_score,_ = Cider().compute_score(gts, res)
    except ImportError:
        cider_score = 0.0
    return {
        "BLEU-1":      round(bleu_n(preds,refs,1), 4),
        "BLEU-4":      round(bleu_n(preds,refs,4), 4),
        "ROUGE-L":     round(rouge_l(preds,refs),  4),
        "CIDEr":       round(float(cider_score),   4),
        "Clinical-F1": round(clin_f1(preds,refs),  4),
    }

print("Evaluation functions defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9 ─ Training & eval functions
# ═══════════════════════════════════════════════════════════════════════════════
import time

def train_epoch(model, encoder, loader, optimizer):
    model.bottleneck.train()
    total = 0
    for i, (uids, vis, inp, attn, lbl) in enumerate(loader):
        B, V, C, H, W = vis.shape
        flat  = vis.view(B*V, C, H, W).to(DEVICE)
        feats = encoder(flat).view(B, V, -1, encoder.hidden_dim).mean(1)
        inp, attn, lbl = inp.to(DEVICE), attn.to(DEVICE), lbl.to(DEVICE)
        optimizer.zero_grad()
        loss = model(feats, inp, attn, lbl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.bottleneck.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
        if (i+1) % 50 == 0:
            print(f"  step {i+1}/{len(loader)}  loss={loss.item():.4f}")
    return total / max(len(loader), 1)

from tqdm.auto import tqdm

@torch.no_grad()
def evaluate_split(model, encoder, loader):
    model.eval()
    preds, refs = [], []
    for uids, vis, inp, attn, lbl in tqdm(loader, desc="  Generating Reports"):
        B, V, C, H, W = vis.shape
        flat  = vis.view(B*V, C, H, W).to(DEVICE)
        feats = encoder(flat).view(B, V, -1, encoder.hidden_dim).mean(1)
        preds.extend(model.generate(feats, max_new_tokens=MAX_TEXT_LEN, beam_size=BEAM_SIZE))
        for l in lbl:
            t = l[l != -100].tolist()
            refs.append(model.tokenizer.decode(t, skip_special_tokens=True))
    return compute_metrics(preds, refs)

print("Training functions defined.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10 ─ Load shared Swin Encoder
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading Swin encoder: {ENCODER_NAME} ...")
encoder = FrozenSwinEncoder(ENCODER_NAME).to(DEVICE)
print(f"✓ Encoder hidden_dim: {encoder.hidden_dim}")

# ── Pre-load tokenizer once to build datasets ─────────────────────────────────
print(f"Loading Llama-3.1-8B tokenizer ...")
_tok = AutoTokenizer.from_pretrained(LLM_NAME)
if _tok.pad_token is None:
    _tok.pad_token = _tok.eos_token
print("✓ Tokenizer ready.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11 ─ Build DataLoaders
# ═══════════════════════════════════════════════════════════════════════════════
train_ds = IUXrayDataset("train", _tok, train_tf)
val_ds   = IUXrayDataset("val",   _tok, val_tf)
test_ds  = IUXrayDataset("test",  _tok, val_tf)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate, num_workers=2)
val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=2)
test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=2)

print(f"✓ Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12 ─ Train the selected bottleneck
# ═══════════════════════════════════════════════════════════════════════════════
import csv

all_results = {}
bt = CURRENT_BOTTLENECK

print(f"\n{'='*60}")
print(f"  TRAINING: {bt.upper()} BOTTLENECK  (Llama-3.1-8B backbone)")
print(f"{'='*60}")

model = QCXRLlama3Model(LLM_NAME, bt)
model.bottleneck = model.bottleneck.to(DEVICE)

trainable = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_bleu4 = 0.0
ckpt = RESULTS / f"best_{bt}_llama3.pt"

for epoch in range(1, EPOCHS + 1):
    t0   = time.time()
    loss = train_epoch(model, encoder, train_loader, optimizer)
    scheduler.step()
    vm   = evaluate_split(model, encoder, val_loader)
    print(f"  Epoch {epoch}/{EPOCHS}  loss={loss:.4f}  val={vm}  ({time.time()-t0:.0f}s)")
    if vm["BLEU-4"] > best_bleu4:
        best_bleu4 = vm["BLEU-4"]
        torch.save(model.bottleneck.state_dict(), ckpt)
        print(f"  *** Best BLEU-4={best_bleu4:.4f} saved → {ckpt.name} ***")

# ── Test evaluation ───────────────────────────────────────────────────────
model.bottleneck.load_state_dict(torch.load(ckpt))
test_m = evaluate_split(model, encoder, test_loader)
print(f"\n  TEST RESULTS [{bt}]: {test_m}")
all_results[bt] = test_m

print("\n✓ All 3 bottlenecks trained!")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13 ─ Final results table
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  FINAL RESULTS TABLE  (Llama-3.1-8B backbone)")
print(f"{'='*65}")
print(f"{'Model':<22} {'BLEU':>8} {'ROUGE':>8} {'CIDEr':>8} {'Clin-F1':>9}")
print("-" * 65)
for name, m in all_results.items():
    print(
        f"{name:<22} {m['BLEU-4']:>8.4f} {m['ROUGE-L']:>8.4f} "
        f"{m['CIDEr']:>8.4f} {m['Clinical-F1']:>9.4f}"
    )
print("=" * 65)

print("\nPaper targets (QCXR-Flamingo Table 1):")
targets = [
    ("Linear",         0.25, 0.30, 0.80, 0.60),
    ("MLP",            0.28, 0.33, 0.90, 0.65),
    ("Transformer",    0.30, 0.35, 1.00, 0.68),
    ("QCXR-Flamingo",  0.31, 0.36, 1.05, 0.70),
]
for row in targets:
    print(f"  {row[0]:<16} BLEU≈{row[1]}  ROUGE≈{row[2]}  CIDEr≈{row[3]}  Clin-F1≈{row[4]}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_path = RESULTS / "metrics_llama3.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(
        f, fieldnames=["Model", "BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr", "Clinical-F1"]
    )
    w.writeheader()
    for name, m in all_results.items():
        w.writerow({"Model": name, **m})
print(f"\n✓ CSV saved → {csv_path}")
