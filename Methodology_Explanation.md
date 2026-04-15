# Methodology: What Exactly Are We Doing?

When building QCXR-Flamingo, it is critical to understand the precise machine learning paradigm we are using. Are we fine-tuning an LLM? Are we training from scratch? Why are we freezing parts of the network? This document answers these theoretical questions.

---

## 1. What exactly are we doing?

We are performing **Visual Parameter-Efficient Fine-Tuning (PEFT)**, specifically utilizing a technique known as **Prefix Tuning** (or Visual Prompt Tuning).

Instead of treating Image Analysis and Text Generation as two separate tasks, we are tricking a Large Language Model (LLM) into "seeing." We do this by:
1. Pushing an X-Ray completely through a Vision Encoder (Swin Transformer) to get raw visual numbers (features).
2. Pushing those numbers through a tiny **Bottleneck** (our Linear/MLP/Transformer/Quantum block).
3. The Bottleneck translates those visual numbers into a "Word Embedding" — a mathematical format that the LLM understands as if it were a typed word.
4. We essentially "type" this fake visual word as a prefix into the LLM (e.g., `[Visual_Token] [BOS] Generate a report...`). 
5. The LLM then simply auto-completes the sentence, effectively writing the radiology report.

---

## 2. Is this LLM Fine-Tuning?

**Strictly speaking: No.** We are absolutely *not* fine-tuning the LLM. 

In our code, both the **Swin Encoder** and the **LLaMA-3 LLM** are 100% frozen. Their internal weights (the billions of parameters that dictate how they understand English or see edges) are completely locked. No gradients are passed into them, and they do not change during our 15 Epochs of training.

**We are ONLY training the Bottleneck.** 
When the LLM makes a mistake writing the report, the error (Loss) flows backward through the frozen LLM, but we only apply updates (Optimizations) to the ~2 million parameters in the Bottleneck. We are strictly training the "Translator," not the "Brain" (LLM) or the "Eyes" (Swin).

---

## 3. Is it acceptable to use frozen LLMs for academic papers?

**Yes — in fact, it is the current Gold Standard.**

Almost all top-tier Vision-Language Model (VLM) research published in the last two years specifically relies on frozen LLMs. Notable paradigm-defining papers include:
*   **Flamingo (DeepMind):** Froze the Chinchilla language model and only trained cross-attention injection layers.
*   **BLIP-2 (Salesforce):** Froze both the Vision Encoder and the LLM, and solely trained the Q-Former (a bottleneck).
*   **LLaVA**: Froze LLaMA and heavily relied on training just the visual projection linear layers. 
*   **R2GenGPT**: The direct inspiration for our architecture; it successfully published state-of-the-art results on IU-Xray by freezing Vicuna-7B.

### Why do academic papers prefer frozen LLMs?
1. **Avoiding Catastrophic Forgetting:** LLaMA-3-8B spent millions of dollars reading the entire internet to understand grammar, biological terms, and logic. If you fine-tune its internal weights on just 4,000 X-ray reports, it will rapidly "forget" its superior general knowledge and overfit to the narrow grammar of IU-Xray. Freezing it preserves its massive intelligence.
2. **Computational Viability:** Fully fine-tuning an 8 Billion parameter model requires massive compute clusters (e.g., 8x A100 80GB GPUs). By freezing it, we can train the network on a single free Kaggle T4 GPU. The academic community heavily values architectures that achieve high results without requiring million-dollar compute budgets.
3. **Isolating Your Novelty:** Since the foundation models are standard and frozen, any performance improvement in the system is directly mathematically attributable to **your specific Bottleneck**. When you publish the paper, establishing that the VQC (Quantum Bottleneck) outperforms the Classical Bottleneck is exceptionally clean, because the LLM wasn't changing and altering the variables. All credit goes to the quantum module!
