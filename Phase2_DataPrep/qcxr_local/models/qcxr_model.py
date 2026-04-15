"""
models/qcxr_model.py
Full QCXR-Flamingo pipeline (baseline variants):
  Frozen Swin → [cached features] → Trainable Bottleneck → Frozen LLM → Loss

The LLM is a causal language model (DistilGPT2 locally, TinyLlama on Kaggle).
The visual token is prepended to the text embeddings — exact Flamingo approach.
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .bottleneck import get_bottleneck


class QCXRModel(nn.Module):
    """
    QCXR-Flamingo model (classical bottleneck baselines).

    Args:
        llm_name       : HuggingFace model id (e.g. 'distilgpt2')
        bottleneck_name: 'linear' | 'mlp' | 'transformer'
        visual_dim     : hidden dim from Swin encoder
        nhead          : heads for transformer bottleneck
        trans_layers   : layers for transformer bottleneck
    """

    def __init__(self, llm_name: str, bottleneck_name: str,
                 visual_dim: int,
                 nhead: int = 8, trans_layers: int = 2):
        super().__init__()

        # ── Frozen LLM ────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        for p in self.llm.parameters():
            p.requires_grad_(False)
        self.llm.eval()

        llm_dim = self.llm.config.hidden_size   # auto-detect from model config

        # ── Trainable bottleneck ──────────────────────────────────────────────
        self.bottleneck = get_bottleneck(
            bottleneck_name, visual_dim, llm_dim, nhead, trans_layers
        )

        self.llm_dim      = llm_dim
        self.bottleneck_name = bottleneck_name

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_embed_layer(self):
        """Return the token embedding layer regardless of LLM architecture."""
        if hasattr(self.llm, "transformer"):          # GPT-2 / DistilGPT-2
            return self.llm.transformer.wte
        if hasattr(self.llm, "model"):                # LLaMA / TinyLlama / Mistral
            return self.llm.model.embed_tokens
        raise AttributeError("Unknown LLM architecture – update _get_embed_layer()")

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self,
                visual_features: torch.Tensor,   # [B, N, visual_dim]  (pre-cached or live)
                input_ids: torch.Tensor,          # [B, T]
                attention_mask: torch.Tensor,     # [B, T]
                labels: torch.Tensor):            # [B, T]  (-100 for non-report tokens)
        """
        Returns CausalLM loss (scalar).
        """
        # 1. Visual token(s) via bottleneck
        vis_token = self.bottleneck(visual_features)   # [B, 1, llm_dim]

        # 2. Text embeddings from frozen LLM
        embed = self._get_embed_layer()
        text_embeds = embed(input_ids)                 # [B, T, llm_dim]

        # 3. Prepend visual token → [B, 1+T, llm_dim]
        inputs_embeds = torch.cat([vis_token, text_embeds], dim=1)

        # 4. Extend attention_mask and labels for the prepended visual token
        B = input_ids.size(0)
        vis_mask   = torch.ones(B, 1, device=input_ids.device, dtype=attention_mask.dtype)
        vis_labels = torch.full((B, 1), -100, device=input_ids.device, dtype=labels.dtype)

        attn   = torch.cat([vis_mask, attention_mask], dim=1)    # [B, 1+T]
        labels_ = torch.cat([vis_labels, labels], dim=1)          # [B, 1+T]

        # 5. LLM forward (frozen)
        out = self.llm(inputs_embeds=inputs_embeds,
                       attention_mask=attn,
                       labels=labels_)
        return out.loss

    # ── Generation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, visual_features: torch.Tensor,
                 max_new_tokens: int = 80, beam_size: int = 3) -> list[str]:
        """
        Greedy / beam-search generation.
        Returns: list of decoded report strings.
        """
        B = visual_features.size(0)
        vis_token = self.bottleneck(visual_features)   # [B, 1, llm_dim]

        # BOS token as starting input
        bos = torch.full((B, 1), self.tokenizer.bos_token_id or 0,
                         device=visual_features.device, dtype=torch.long)
        embed = self._get_embed_layer()
        bos_embed = embed(bos)                         # [B, 1, llm_dim]

        inputs_embeds = torch.cat([vis_token, bos_embed], dim=1)
        attn = torch.ones(B, inputs_embeds.size(1),
                          device=visual_features.device, dtype=torch.long)

        generated = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
