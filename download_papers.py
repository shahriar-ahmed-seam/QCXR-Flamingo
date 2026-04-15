"""
Literature Survey Paper Downloader
Downloads all available free papers from the QCXR-Flamingo literature survey.
Papers are saved into: Papers_Downloaded/<Venue>/<filename>.pdf
"""

import os
import time
import requests
from pathlib import Path

# ─────────────────────────────────────────────────
# PAPER LIST: (Title, Venue+Year, PDF_URL)
# Sources: arXiv, MICCAI Open Access, CVF, IJCAI preprints
# ─────────────────────────────────────────────────
PAPERS = [
    # ── Our 2 Reference Papers ──────────────────────────────────────────────
    (
        "R2GenGPT_Radiology_Report_Generation_with_Frozen_LLMs",
        "Reference_Papers",
        "https://arxiv.org/pdf/2309.09812"
    ),

    # ── CVPR 2024 ────────────────────────────────────────────────────────────
    (
        "InVERGe_Intelligent_Visual_Encoder_Bridging_Modalities_CVPR2024",
        "CVPR_2024",
        "https://openaccess.thecvf.com/content/CVPR2024/papers/Roy_InVERGe_Intelligent_Visual_Encoder_for_Bridging_Modalities_in_Report_Generation_CVPR_2024_paper.pdf"
    ),

    # ── CVPR 2025 ────────────────────────────────────────────────────────────
    (
        "Enhanced_Contrastive_Learning_MultiView_Longitudinal_CXR_CVPR2025",
        "CVPR_2025",
        "https://arxiv.org/pdf/2502.20056"
    ),
    (
        "FactCheXcker_Mitigating_Measurement_Hallucinations_CXR_CVPR2025",
        "CVPR_2025",
        "https://arxiv.org/pdf/2411.18672"
    ),
    (
        "CXPMRG_Bench_Pretraining_Benchmarking_Xray_CVPR2025",
        "CVPR_2025",
        "https://arxiv.org/pdf/2410.00379"
    ),
    (
        "VILA-M3_Enhancing_VLMs_with_Medical_Expert_Knowledge_CVPR2025",
        "CVPR_2025",
        "https://arxiv.org/pdf/2411.12915"
    ),

    # ── CVPR 2026 ────────────────────────────────────────────────────────────
    (
        "X-WIN_Chest_Radiograph_World_Model_CVPR2026",
        "CVPR_2026",
        "https://arxiv.org/pdf/2511.14918"
    ),

    # ── ICLR 2024 ────────────────────────────────────────────────────────────
    (
        "BioBridge_Bridging_Biomedical_Foundation_Models_ICLR2024",
        "ICLR_2024",
        "https://arxiv.org/pdf/2310.03320"
    ),

    # ── ICLR 2025 ────────────────────────────────────────────────────────────
    (
        "MedRegA_Interpretable_Bilingual_Multimodal_LLM_ICLR2025",
        "ICLR_2025",
        "https://arxiv.org/pdf/2410.18387"
    ),

    # ── ICLR 2026 ────────────────────────────────────────────────────────────
    (
        "Learning_Self-Critiquing_Mechanisms_Region_Guided_CXR_ICLR2026",
        "ICLR_2026",
        "https://openreview.net/pdf?id=6sOSwgCmpH"
    ),
    (
        "Rethinking_Radiology_Report_Generation_Topic_Guided_ICLR2026",
        "ICLR_2026",
        "https://openreview.net/pdf?id=nV3SAjFlyv"
    ),

    # ── ICCV 2025 ────────────────────────────────────────────────────────────
    (
        "GEMeX_Large_Scale_Groundable_Explainable_Medical_VQA_ICCV2025",
        "ICCV_2025",
        "https://arxiv.org/pdf/2411.16778"
    ),

    # ── NeurIPS 2025 ─────────────────────────────────────────────────────────
    (
        "CURV_Coherent_Uncertainty_Aware_Reasoning_VLMs_CXR_NeurIPS2025",
        "NeurIPS_2025",
        "https://arxiv.org/pdf/2504.07416"  # RadZero/related arXiv
    ),

    # ── AAAI 2025 ────────────────────────────────────────────────────────────
    (
        "Radiology_Report_Multi-objective_Preference_Optimization_AAAI2025",
        "AAAI_2025",
        "https://arxiv.org/pdf/2412.08901"
    ),
    (
        "HC-LLM_Historical_Constrained_LLM_Radiology_AAAI2025",
        "AAAI_2025",
        "https://arxiv.org/pdf/2412.11070"
    ),
    (
        "LLM-RG4_Flexible_Factual_Radiology_Report_AAAI2025",
        "AAAI_2025",
        "https://arxiv.org/pdf/2412.12001"
    ),

    # ── AAAI 2026 ────────────────────────────────────────────────────────────
    (
        "PriorRG_Prior_Guided_Contrastive_Pretraining_CXR_AAAI2026",
        "AAAI_2026",
        "https://arxiv.org/pdf/2508.05353"
    ),
    (
        "S2D-ALIGN_Shallow_to_Deep_Auxiliary_Learning_AAAI2026",
        "AAAI_2026",
        "https://arxiv.org/pdf/2511.11066"
    ),
    (
        "Disease_Aware_Dual_Stage_Framework_CXR_AAAI2026",
        "AAAI_2026",
        "https://arxiv.org/pdf/2511.12259"
    ),

    # ── ICML 2025 ────────────────────────────────────────────────────────────
    (
        "MedRAX_Medical_Reasoning_Agent_Chest_Xray_ICML2025",
        "ICML_2025",
        "https://arxiv.org/pdf/2502.02673"
    ),

    # ── MICCAI 2024 ──────────────────────────────────────────────────────────
    (
        "SEI_Structural_Entities_Extraction_Patient_Indications_MICCAI2024",
        "MICCAI_2024",
        "https://papers.miccai.org/miccai-2024/paper/1768_paper.pdf"
    ),
    (
        "CXRL_Text_Driven_CXR_Generation_Reinforcement_Learning_MICCAI2024",
        "MICCAI_2024",
        "https://papers.miccai.org/miccai-2024/paper/0165_paper.pdf"
    ),
    (
        "ECRG_Energy_Based_Controllable_Radiology_Report_MICCAI2024",
        "MICCAI_2024",
        "https://papers.miccai.org/miccai-2024/paper/0765_paper.pdf"
    ),

    # ── ISBI 2025 ────────────────────────────────────────────────────────────
    (
        "R2Gen-Mamba_Selective_State_Space_Model_Radiology_ISBI2025",
        "ISBI_2025",
        "https://arxiv.org/pdf/2410.18135"
    ),

    # ── IJCAI 2025 ───────────────────────────────────────────────────────────
    (
        "RRG-Mamba_Efficient_Radiology_Report_SSM_IJCAI2025",
        "IJCAI_2025",
        "https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/4573.pdf"
    ),
    (
        "Cyclic_Vision_Language_Manipulator_Radiology_IJCAI2025",
        "IJCAI_2025",
        "https://ijcai-preprints.s3.us-west-1.amazonaws.com/2025/1579.pdf"
    ),

    # ── ACL 2025 ─────────────────────────────────────────────────────────────
    (
        "RADAR_Enhancing_Radiology_Report_Knowledge_Injection_ACL2025",
        "ACL_2025",
        "https://arxiv.org/pdf/2505.14318"
    ),
    (
        "Libra_Leveraging_Temporal_Images_Biomedical_Radiology_ACL2025",
        "ACL_2025",
        "https://arxiv.org/pdf/2411.19378"
    ),

    # ── arXiv / Foundation Models ─────────────────────────────────────────────
    (
        "CheXagent_Foundation_Model_Chest_Xray_Interpretation_arXiv2024",
        "arXiv_2024",
        "https://arxiv.org/pdf/2401.12208"
    ),
    (
        "XrayGPT_Chest_Radiographs_Summarization_LargeMedical_VLMs_ACLW2024",
        "ACLW_2024",
        "https://arxiv.org/pdf/2306.07971"
    ),
    (
        "CoGaze_Context_Gaze_Guided_VLP_Chest_Xray_arXiv2026",
        "arXiv_2026",
        "https://arxiv.org/pdf/2603.26049"
    ),
    (
        "CheXOne_Reasoning_Enabled_VLM_CXR_Interpretation_arXiv2026",
        "arXiv_2026",
        "https://arxiv.org/pdf/2604.00493"
    ),

    # ── Journals (TMI, TIP, ESWA) ────────────────────────────────────────────
    (
        "STREAM_SpatioTemporal_RetrievalAugmented_CXR_TMI2025",
        "TMI_2025",
        "https://arxiv.org/pdf/2311.03140"  # preprint version
    ),
    (
        "CMCRL_CrossModal_Causal_Representation_Learning_TIP2025",
        "TIP_2025",
        "https://arxiv.org/pdf/2404.06798"
    ),
    (
        "HKRG_Hierarchical_Knowledge_Integration_Radiology_ESWA2025",
        "ESWA_2025",
        "https://arxiv.org/pdf/2408.00374"  # likely preprint ID
    ),
]

# ─────────────────────────────────────────────────
# DOWNLOADER
# ─────────────────────────────────────────────────
BASE_DIR = Path(r"c:\Users\Seam\Desktop\Research\NSU\Papers_Downloaded")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def download_paper(filename, venue, url):
    folder = BASE_DIR / venue
    folder.mkdir(parents=True, exist_ok=True)
    filepath = folder / f"{filename}.pdf"

    if filepath.exists():
        print(f"  [SKIP]   Already exists: {filename}.pdf")
        return "skipped"

    try:
        r = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        if r.status_code == 200 and len(r.content) > 5000:
            # Check it's actually a PDF
            if r.content[:4] == b'%PDF' or 'pdf' in r.headers.get('Content-Type', ''):
                filepath.write_bytes(r.content)
                size_kb = len(r.content) / 1024
                print(f"  [OK]     {filename}.pdf  ({size_kb:.0f} KB)")
                return "success"
            else:
                print(f"  [SKIP]   Not a PDF (HTML page?): {url}")
                return "not_pdf"
        else:
            print(f"  [FAIL]   HTTP {r.status_code}: {url}")
            return "failed"
    except Exception as e:
        print(f"  [ERROR]  {filename}: {e}")
        return "error"


def main():
    print("=" * 65)
    print("  QCXR-Flamingo Literature Survey — Paper Downloader")
    print(f"  Output: {BASE_DIR}")
    print("=" * 65)

    results = {"success": 0, "skipped": 0, "failed": 0, "not_pdf": 0, "error": 0}

    for i, (name, venue, url) in enumerate(PAPERS, 1):
        print(f"\n[{i:02d}/{len(PAPERS)}] {name[:55]}...")
        status = download_paper(name, venue, url)
        results[status] = results.get(status, 0) + 1
        if status == "success":
            time.sleep(1.5)  # be polite to servers

    print("\n" + "=" * 65)
    print("  DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"  ✅ Downloaded : {results['success']}")
    print(f"  ⏭️  Skipped    : {results['skipped']}")
    print(f"  ❌ Failed     : {results['failed']}")
    print(f"  ⚠️  Not PDF   : {results['not_pdf']}")
    print(f"  🔥 Errors     : {results['error']}")
    print(f"\n  Papers saved to: {BASE_DIR}")

    # List what was actually downloaded
    print("\n  Downloaded files:")
    for folder in sorted(BASE_DIR.iterdir()):
        if folder.is_dir():
            pdfs = list(folder.glob("*.pdf"))
            if pdfs:
                print(f"\n  📁 {folder.name}/")
                for pdf in sorted(pdfs):
                    size_kb = pdf.stat().st_size / 1024
                    print(f"     📄 {pdf.name}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
