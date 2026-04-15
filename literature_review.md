# Literature Review: Chest X-ray Report Generation in the Age of Multimodal Foundation Models (2024–2026)

## 1. Introduction

Chest X-ray (CXR) report generation is a critical intersection of computer vision and natural language processing in the medical domain. Its goal is to alleviate the significant workload on radiologists by automatically drafting accurate, clinically grounded reports from medical imaging. While early models (e.g., standard CNN-RNN architectures) established basic feasibility, the 2024–2026 era has marked a paradigm shift.

Based on a systematic review of 34 recent papers from top-tier venues (CVPR, AAAI, MICCAI, ICLR, NeurIPS, and leading journals), current research trends divide into several distinct streams:
1. **The transition to Multimodal Large Language Models (MLLMs)** and foundation models.
2. **Innovative lightweight architectures** (such as State Space Models).
3. **Enhanced diagnostic realism** via patient progression tracking, knowledge graph integration, and spatial grounding.
4. **Error mitigation**, primarily addressing clinical hallucination and factual drift.

This review synthesizes the state-of-the-art developments, situating them within the context of next-generation efficiency—a critical jumping-off point for hybrid quantum-classical frameworks like **QCXR-Flamingo**.

---

## 2. From Task-Specific Models to Foundation MLLMs

A significant portion of recent research focuses on adapting generalist LLMs (like LLaMA and Vicuna) or creating domain-specific medical foundation models to act as the cognitive engine for draft generation. 

**Foundation Models and Adaptations:**
- **CheXagent** (arXiv 2024) and **CheXOne** (arXiv 2026) attempt to build zero-shot capable models pre-trained on massive clinical datasets. CheXOne notably incorporates explicit "reasoning traces," linking visual evidence to radiographic findings before producing a final report.
- **XrayGPT** (ACLW 2024) utilizes a simple linear transformation to align a medical visual encoder (MedClip) with a frozen Vicuna LLM, mimicking the Flamingo approach. It relies on 217k interactive summaries to fine-tune the LLM, showcasing the power of visual-instruction tuning.
- **VILA-M3** (CVPR 2025) argues that generalist architectures are not enough. It introduces a specialized Instruction Fine-Tuning (IFT) stage that ingests data from domain-expert models (e.g., targeted classification and segmentation networks) to improve upon generalized models like Med-Gemini.

**Addressing the Bottleneck:**
Central to bridging the visual (scan) and text (LLM) domains is the *visual bottleneck* or *mapper*. The baseline **R2GenGPT** (Wang et al.) demonstrated that a lightweight linear layer could effectively map frozen Swin-Transformer features into a frozen LLaMA-2 model space. Similarly, **InVERGe** (CVPR 2024) employs an intelligent Cross-Modal Query Fusion Layer to optimize this mapping. 

> [!NOTE]
> *Relevance to QCXR-Flamingo:* The industry is solidifying around the "Frozen Encoder + Frozen LLM + Trainable Mapper" paradigm. QCXR-Flamingo's proposal to replace this classical mapper with a Variational Quantum Circuit (VQC) directly targets the heart of this modern architecture, promising severe parameter reduction within the bottleneck layer.

---

## 3. Grounding and Reducing Hallucinations

A persistent danger with utilizing LLMs for clinical reports is "hallucination"—the generation of text that is grammatically flawless but clinically incorrect or ungrounded in the image.

**Factuality and Alignment:**
- **FactCheXcker** (CVPR 2025) attacks measurement hallucinations (e.g., endotracheal tube placement sizes) via an update paradigm that leverages code-generation to verify measurements before adding them to the report.
- Models like **LLM-RG4** (AAAI 2025) focus on minimizing "input-agnostic" hallucinations by incorporating a token-level loss weighting strategy that penalizes generic, ungrounded declarative statements.
- **MPO (Multi-objective Preference Optimization)** (AAAI 2025) borrows RLHF concepts from standard LLMs. Since human radiologists prioritize different metrics (some prefer fluency, others strict diagnostic brevity), MPO aligns the generation with multi-dimensional human preference vectors via reinforcement learning.

**Gaze and Anatomy Grounding:**
To mimic human radiologists—who visually scan specific anatomical regions before writing—researchers are forcing models to adopt region-centric reasoning:
- **MedRegA** (ICLR 2025) and **S2D-ALIGN** (AAAI 2026) leverage regional grounding. S2D-ALIGN implements a shallow-to-deep auxiliary learning strategy that forces the generation to ground itself in specific anatomical key phrases, moving away from simple instance-level image-text pairing.
- **CoGaze** (arXiv 2026) explicitly uses probabilistic priors derived from actual radiologist eye-tracking data to guide the attention modules toward diagnostically salient regions.
- **LLaVA-TA** (ICLR 2026) identifies a flaw in narrative training: models learn linguistic flow at the expense of visual evidence. Their solution dismantles standard narrative flow into independent, anatomically-grounded discrete topics.

---

## 4. Longitudinal Progression and Multi-View Alignment

A static chest X-ray is often only part of the puzzle; real-world patient care relies on analyzing the change between current and previous scans (longitudinal progression).

- **HC-LLM** (AAAI 2025) empowers LLMs with the ability to digest both time-shared and time-specific features from longitudinal images to explicitly state disease progression in the generated text.
- **PriorRG** (AAAI 2026) incorporates patient-specific prior knowledge via a coarse-to-fine decoding phase, demonstrating that referencing an older scan explicitly improves the clinical utility of the report.
- **Libra** (ACL 2025) introduces a Temporal Alignment Connector (TAC) integrated into the vision-language architecture specifically to capture temporal differences between current and prior studies.
- Moving beyond 2D, the **X-WIN** World Model (CVPR 2026) distills volumetric knowledge from 3D CT scans to better predict structural overlap issues common in 2D CXRs, allowing better diagnostic feature extraction.

---

## 5. Efficient Architectures: State Space Models 

While LLMs are powerful, their quadratic attention complexity presents massive computational bottlenecks, especially for high-resolution medical sequences.

2024-2025 marks the rise of specialized efficient architectures bridging the gap:
- **RRG-Mamba** (IJCAI 2025) integrates rotary position encoding (RoPE) into the Mamba State Space Model (SSM) to allow linear-complexity dependency modeling for visual features. 
- **R2Gen-Mamba** (ISBI 2025) showcases that hybrid Transformer-Mamba workflows can process long visual sequences far more efficiently than standard self-attention mechanisms without sacrificing report quality metrics like BLEU or CIDEr.

> [!TIP]
> *Relevance to QCXR-Flamingo:* The integration of Mamba strongly justifies the community's hunger for *computational efficiency* in RRG. QCXR-Flamingo attacks this same efficiency problem using Quantum Machine Learning (QML) rather than SSMs.

---

## 6. Synthesis and Project Relevance

The review of 2024–2026 literature unequivocally proves the validity of the Flamingo-style architecture in modern medical imaging. The current state-of-the-art methodology can be summarized as:
1. Take a powerful, pre-trained visual encoder (Swin/ViT).
2. Take a robust, instruction-tuned LLM (LLaMA/Vicuna).
3. Connect them via a mapping bottleneck, integrating domain knowledge (RAG, Knowledge Graphs) and controlling generation via preference optimization.

**Conclusion for QCXR-Flamingo:**
The classical mapping layer is currently the most heavily engineered component in modern architectures, burdened with multi-view alignment, longitudinal analysis, and region-grounding. By introducing a **Variational Quantum Circuit (VQC)** to replace this classical mapper, QCXR-Flamingo steps directly into the frontier of resource-efficient medical AI. 

To ensure the QCXR-Flamingo project is benchmarked robustly against top-tier standards, it must be compared against the standard classical linear mappers used in **R2GenGPT** and ideally evaluated on clinical accuracy metrics (like clinical F1 / CheXpert F1-14) alongside standard NLP metrics (BLEU/ROUGE), directly addressing the field's current focus on reducing clinical hallucination.
