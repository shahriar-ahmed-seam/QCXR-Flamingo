# Extracted Abstracts and Key Info
## HC-LLM_Historical_Constrained_LLM_Radiology_AAAI2025.pdf
**Abstract/Intro Start:**
Radiology report generation (RRG) models typically focus
on individual exams, often overlooking the integration of
historical visual or textual data, which is crucial for patient
follow-ups. Traditional methods usually struggle with long
sequence dependencies when incorporating historical infor-
mation, but large language models (LLMs) excel at in-context
learning, making them well-suited for analyzing longitudinal
medical data. In light of this, we propose a novel Historical-
Constrained Large Language Models (HC-LLM) framework
for RRG, empowering LLMs with longitudinal report gen-
eration capabilities by constraining the consistency and dif-
ferences between longitudinal images and their correspond-
ing reports. Specifically, our approach extracts both time-
shared and time-specific features from longitudinal chest X-
rays and diagnostic reports to capture disease progression.
Then, we ensure consistent representation by applying intra-
modality similarity constraints and aligning various features
across modalities with multimodal contrastive and structural
constraints. These combined constraints effectively guide the
LLMs in generating diagnostic reports that accurately re-
flect the progression of the disease, achieving state-of-the-
art results on the Longitudinal-MIMIC dataset. Notably, our
approach performs well even without historical data dur-
ing testing and can be easily adapted to other multimodal
large models, enhancing its versatility. Code is available at:
https://github.com/TengfeiLiu966/HC-LLM.

---
## LLM-RG4_Flexible_Factual_Radiology_Report_AAAI2025.pdf
**Abstract/Intro Start:**
Drafting radiology reports is a complex task requiring flex-
ibility, where radiologists tail content to available informa-
tion and particular clinical demands. However, most current
radiology report generation (RRG) models are constrained
to a fixed task paradigm, such as predicting the full “find-
ing” section from a single image, inherently involving a mis-
match between inputs and outputs. The trained models lack
the flexibility for diverse inputs and could generate harm-
ful, input-agnostic hallucinations. To bridge the gap between
current RRG models and the clinical demands in practice,
we first develop a data generation pipeline to create a new
MIMIC-RG4 dataset, which considers four common radiol-
ogy report drafting scenarios and has perfectly corresponded
input and output. Secondly, we propose a novel large lan-
guage model (LLM) based RRG framework, namely LLM-
RG4, which utilizes LLM’s flexible instruction-following ca-
pabilities and extensive general knowledge. We further de-
velop an adaptive token fusion module that offers flexibility
to handle diverse scenarios with different input combinations,
while minimizing the additional computational burden asso-
ciated with increased input volumes. Besides, we propose a
token-level loss weighting strategy to direct the model’s at-
tention towards positive and uncertain descriptions. Exper-
imental results demonstrate that LLM-RG4 achieves state-
of-the-art performance in both clinical efficiency and natural
language generation on the MIMIC-RG4 and MIMIC-CXR
datasets. We quantitatively demonstrate that our model has
minimal input-agnostic hallucinations, whereas current open-
source models commonly suffer from this problem.
Code — https://github.com/zh-Wang-Med/LLM-RG4

---
## Radiology_Report_Multi-objective_Preference_Optimization_AAAI2025.pdf
**Abstract/Intro Start:**
Automatic Radiology Report Generation (RRG) is an impor-
tant topic for alleviating the substantial workload of radiolo-
gists. Existing RRG approaches rely on supervised regression
based on different architectures or additional knowledge in-
jection, while the generated report may not align optimally
with radiologists’ preferences. Especially, since the prefer-
ences of radiologists are inherently heterogeneous and multi-
dimensional, e.g., some may prioritize report fluency, while
others emphasize clinical accuracy. To address this problem,
we propose a new RRG method via Multi-objective Pref-
erence Optimization (MPO) to align the pre-trained RRG
model with multiple human preferences, which can be formu-
lated by multi-dimensional reward functions and optimized
by multi-objective reinforcement learning (RL). Specifically,
we use a preference vector to represent the weight of pref-
erences and use it as a condition for the RRG model. Then,
a linearly weighed reward is obtained via a dot product be-
tween the preference vector and multi-dimensional reward.
Next, the RRG model is optimized to align with the prefer-
ence vector by optimizing such a reward via RL. In the train-
ing stage, we randomly sample diverse preference vectors
from the preference space and align the model by optimizing
the weighted multi-objective rewards, which leads to an op-
timal policy on the entire preference space. When inference,
our model can generate reports aligned with specific prefer-
ences without further fine-tuning. Extensive experiments on
two public datasets show the proposed method can generate
reports that cater to different preferences in a single model
and achieve state-of-the-art performance.

---
## Disease_Aware_Dual_Stage_Framework_CXR_AAAI2026.pdf
**Abstract/Intro Start:**
Radiology report generation from chest X-rays is an important task in artificial intelligence with
the potential to greatly reduce radiologists’ workload and shorten patient wait times. Despite
recent advances, existing approaches often lack sufficient disease-awareness in visual represen-
tations and adequate vision-language alignment to meet the specialized requirements of medical
image analysis. As a result, these models usually overlook critical pathological features on chest
X-rays and struggle to generate clinically accurate reports. To address these limitations, we pro-
pose a novel dual-stage disease-aware framework for chest X-ray report generation. In Stage 1,
our model learns Disease-Aware Semantic Tokens (DASTs) corresponding to specific pathology
categories through cross-attention mechanisms and multi-label classification, while simultane-
ously aligning vision and language representations via contrastive learning. In Stage 2, we in-
troduce a Disease-Visual Attention Fusion (DVAF) module to integrate disease-aware represen-
tations with visual features, along with a Dual-Modal Similarity Retrieval (DMSR) mechanism
that combines visual and disease-specific similarities to retrieve relevant exemplars, providing
contextual guidance during report generation. Extensive experiments on benchmark datasets
(i.e., CheXpert Plus, IU X-ray, and MIMIC-CXR) demonstrate that our disease-aware frame-
work achieves state-of-the-art performance in chest X-ray report generation, with significant
improvements in clinical accuracy and linguistic quality.

---
## PriorRG_Prior_Guided_Contrastive_Pretraining_CXR_AAAI2026.pdf
**Abstract/Intro Start:**
Chest X-ray report generation aims to reduce radiologists’
workload by automatically producing high-quality prelimi-
nary reports. A critical yet underexplored aspect of this task
is the effective use of patient-specific prior knowledge—
including clinical context (e.g., symptoms, medical history)
and the most recent prior image—which radiologists rou-
tinely rely on for diagnostic reasoning. Most existing methods
generate reports from single images, neglecting this essential
prior information and thus failing to capture diagnostic in-
tent or disease progression. To bridge this gap, we propose
PriorRG, a novel chest X-ray report generation framework
that emulates real-world clinical workflows via a two-stage
training pipeline. In Stage 1, we introduce a prior-guided con-
trastive pre-training scheme that leverages clinical context to
guide spatiotemporal feature extraction, allowing the model
to align more closely with the intrinsic spatiotemporal seman-
tics in radiology reports. In Stage 2, we present a prior-aware
coarse-to-fine decoding for report generation that progres-
sively integrates patient-specific prior knowledge with the vi-
sion encoder’s hidden states. This decoding allows the model
to align with diagnostic focus and track disease progression,
thereby enhancing the clinical accuracy and fluency of the
generated reports. Extensive experiments on MIMIC-CXR
and MIMIC-ABN datasets demonstrate that PriorRG out-
performs state-of-the-art methods, achieving a 3.6% BLEU-
4 and 3.8% F1 score improvement on MIMIC-CXR, and a
5.9% BLEU-1 gain on MIMIC-ABN.
Code — https://github.com/mk-runner/PriorRG
Extended version — https://arxiv.org/abs/2508.05353
1

---
## S2D-ALIGN_Shallow_to_Deep_Auxiliary_Learning_AAAI2026.pdf
**Abstract/Intro Start:**
Radiology Report Generation (RRG) aims to automati-
cally generate diagnostic reports from radiology images. To
achieve this, existing methods have leveraged the power-
ful cross-modal generation capabilities of Multimodal Large
Language Models (MLLMs), primarily focusing on opti-
mizing cross-modal alignment between radiographs and re-
ports through Supervised Fine-Tuning (SFT). However, by
only performing instance-level alignment with the image-
text pairs, the standard SFT paradigm fails to establish
anatomically-grounded alignment, where the templated na-
ture of reports often leads to sub-optimal generation qual-
ity. To address this, we propose S2D-ALIGN, a novel SFT
paradigm that establishes anatomically-grounded alignment
by leveraging auxiliary signals of varying granularities.
S2D-ALIGN implements a shallow-to-deep strategy, progres-
sively enriching the alignment process: it begins with the
coarse radiograph-report pairing, then introduces reference
reports for instance-level guidance, and ultimately utilizes
key phrases to ground the generation in specific anatom-
ical details. To bridge the different alignment stages, we
introduce a memory-based adapter that empowers feature
sharing, thereby integrating coarse and fine-grained guid-
ance. For evaluation, we conduct experiments on the pub-
lic MIMIC-CXR and IU X-RAY benchmarks, where S2D-
ALIGN achieves state-of-the-art performance compared to
existing methods. Ablation studies validate the effectiveness
of our multi-stage, auxiliary-guided approach, highlighting a
promising direction for enhancing grounding capabilities in
complex, multi-modal generation tasks.

---
## XrayGPT_Chest_Radiographs_Summarization_LargeMedical_VLMs_ACLW2024.pdf
**Abstract/Intro Start:**
The latest breakthroughs in large vision-
language models, such as Bard and GPT-4,
have showcased extraordinary abilities in per-
forming a wide range of tasks. Such models
are trained on massive datasets comprising bil-
lions of public image-text pairs with diverse
tasks. However, their performance on task-
specific domains, such as radiology, is still
under-investigated and potentially limited due
to a lack of sophistication in understanding
biomedical images. On the other hand, con-
versational medical models have exhibited re-
markable success but have mainly focused on
text-based analysis. In this paper, we intro-
duce XrayGPT, a novel conversational medical
vision-language model that can analyze and
answer open-ended questions about chest ra-
diographs. Specifically, we align both medi-
cal visual encoder (MedClip) with a fine-tuned
large language model (Vicuna), using a sim-
ple linear transformation. This alignment en-
ables our model to possess exceptional visual
conversation abilities, grounded in a deep un-
derstanding of radiographs and medical do-
main knowledge. To enhance the performance
of LLMs in the medical context, we generate
217k interactive and high-quality summaries
from free-text radiology reports. These sum-
maries serve to enhance the performance of
LLMs through the fine-tuning process. Our
approach opens up new avenues the research
for advancing the automated analysis of chest
radiographs. Our open-source demos, models,
and instruction sets are available at: https:
//github.com/mbzuai-oryx/XrayGPT
1

---
## Libra_Leveraging_Temporal_Images_Biomedical_Radiology_ACL2025.pdf
**Abstract/Intro Start:**
Radiology report generation (RRG) requires
advanced medical image analysis, effective
temporal reasoning, and accurate text gener-
ation. While multimodal large language mod-
els (MLLMs) align with pre-trained vision
encoders to enhance visual-language under-
standing, most existing methods rely on single-
image analysis or rule-based heuristics to pro-
cess multiple images, failing to fully leverage
temporal information in multi-modal medical
datasets. In this paper, we introduce Libra, a
temporal-aware MLLM tailored for chest X-ray
report generation. Libra combines a radiology-
specific image encoder with a novel Tempo-
ral Alignment Connector (TAC), designed to
accurately capture and integrate temporal dif-
ferences between paired current and prior im-
ages. Extensive experiments on the MIMIC-
CXR dataset demonstrate that Libra establishes
a new state-of-the-art benchmark among simi-
larly scaled MLLMs, setting new standards in
both clinical relevance and lexical accuracy.
1

---
## RADAR_Enhancing_Radiology_Report_Knowledge_Injection_ACL2025.pdf
**Abstract/Intro Start:**
Large language models (LLMs) have demon-
strated remarkable capabilities in various do-
mains, including radiology report generation.
Previous approaches have attempted to utilize
multimodal LLMs for this task, enhancing their
performance through the integration of domain-
specific knowledge retrieval. However, these
approaches often overlook the knowledge al-
ready embedded within the LLMs, leading to
redundant information integration. To address
this limitation, we propose RADAR, a frame-
work for enhancing radiology report genera-
tion with supplementary knowledge injection.
RADAR improves report generation by system-
atically leveraging both the internal knowledge
of an LLM and externally retrieved informa-
tion. Specifically, it first extracts the model’s ac-
quired knowledge that aligns with expert image-
based classification outputs. It then retrieves
relevant supplementary knowledge to further
enrich this information. Finally, by aggregat-
ing both sources, RADAR generates more accu-
rate and informative radiology reports. Exten-
sive experiments on MIMIC-CXR, CHEXPERT-
PLUS, and IU X-RAY demonstrate that our
model outperforms state-of-the-art LLMs in
both language quality and clinical accuracy1.
1

---
## CheXagent_Foundation_Model_Chest_Xray_Interpretation_arXiv2024.pdf
**Abstract/Intro Start:**
A Vision-Language Foundation Model to Enhance
Efficiency of Chest X-ray Interpretation
Zhihong Chen1,2,∗, Maya Varma1,2,3,∗, Justin Xu1,2,4, Magdalini Paschali1,2, Dave Van Veen1,5,
Andrew Johnston2, Alaa Youssef1,2, Louis Blankemeier1,5, Christian Bluethgen1,6,
Stephan Altmayer2, Jeya Maria Jose Valanarasu1,3, Mohamed Siddig Eltayeb Muneer2,
Eduardo Pontes Reis1,2, Joseph Paul Cohen1, Cameron Olsen2, Tanishq Mathew Abraham7,
Emily B. Tsai2, Christopher F. Beaulieu2, Jenia Jitsev8,9, Sergios Gatidis1,2,
Jean-Benoit Delbrouck1,2, Akshay S. Chaudhari1,2,10, Curtis P. Langlotz1,2,10,11
1Stanford Center for Artificial Intelligence in Medicine and Imaging, Stanford University, Palo Alto, CA, USA.
2Department of Radiology, Stanford University, Stanford, CA, USA. 3Department of Computer Science, Stanford
University, Stanford, CA, USA. 4Big Data Institute, University of Oxford, Oxford, UK. 5Department of Electrical
Engineering, Stanford University, Stanford, CA, USA. 6Department of Radiology, University Hospital Zurich, Zürich,
Switzerland. 7Stability AI, London, UK. 8Jülich Supercomputing Centre, Jülich, Germany. 9LAION, Germany.
10Department of Biomedical Data Science, Stanford University, Stanford, CA, USA. 11Department of Medicine, Stanford
University, Stanford, CA, USA. Corresponding to: {zhihongc,mvarma2,jbdel,akshaysc,langlotz}@stanford.edu
Over 1.4 billion chest X-rays (CXRs) are performed annually due to their cost-effectiveness as an initial
diagnostic test. This scale of radiological studies provides a significant opportunity to streamline CXR
interpretation and documentation. While foundation models are a promising solution, the lack of
publicly available large-scale datasets and benchmarks inhibits their iterative development and real-world
evaluation. To overcome these challenges, we constructed a large-scale dataset (CheXinstruct), which
we utilized to train a vision-language foundation model (CheXagent). We systematically demonstrated
competitive performance

---
## CheXOne_Reasoning_Enabled_VLM_CXR_Interpretation_arXiv2026.pdf
**Abstract/Intro Start:**
Chest X-rays (CXRs) are among the most frequently performed imaging exam-
inations worldwide, yet rising imaging volumes increase radiologist workload
and the risk of diagnostic errors. Although artificial intelligence (AI) systems
have shown promise for CXR interpretation, most generate only final predic-
tions, without making explicit how visual evidence is translated into radiographic
findings and diagnostic predictions. We present CheXOne, a reasoning-enabled
1
arXiv:2604.00493v1  [cs.CV]  1 Apr 2026

vision–language model for CXR interpretation. CheXOne jointly generates diag-
nostic predictions and explicit, clinically grounded reasoning traces that connect
visual evidence, radiographic findings, and these predictions. The model is
trained on 14.7 million instruction and reasoning samples curated from 30 public
datasets spanning 36 CXR interpretation tasks, using a two-stage framework that
combines instruction tuning with reinforcement learning to improve reasoning
quality. We evaluate CheXOne in zero-shot settings across visual question answer-
ing, report generation, visual grounding and reasoning assessment, covering
17 evaluation settings. CheXOne outperforms existing medical and general-
domain foundation models and achieves strong performance on independent
public benchmarks. A clinical reader study demonstrates that CheXOne-drafted
reports are comparable to or better than resident-written reports in 55% of
cases, while effectively addressing clinical indications and enhancing both report
writing and CXR interpretation efficiency. Further analyses involving radiolo-
gists reveal that the generated reasoning traces show high clinical factuality and
provide causal support for the final predictions, offering a plausible explana-
tion for the performance gains. These results suggest that explicit reasoning can
improve model performance, interpretability and clinical utility in AI-assisted
CXR interpretation.
Keywords: Chest X-ray, vision-language model, foundation model, reasoning
1

---
## CoGaze_Context_Gaze_Guided_VLP_Chest_Xray_arXiv2026.pdf
**Abstract/Intro Start:**
Despite recent advances in medical vision-language pretraining,
existing models still struggle to capture the diagnostic workflow: ra-
diographs are typically treated as context-agnostic images, while ra-
diologists’ gaze—a crucial cue for visual reasoning—remains largely
underexplored by existing methods. These limitations hinder the
modeling of disease-specific patterns and weaken cross-modal align-
ment. To bridge this gap, we introduce CoGaze, a Context- and
Gaze-guided vision-language pretraining framework for chest X-
rays. We first propose a context-infused vision encoder that mod-
els how radiologists integrate clinical context—including patient
history, symptoms, and diagnostic intent—to guide diagnostic rea-
soning. We then present a multi-level supervision paradigm that
(1) enforces intra- and inter-modal semantic alignment through
hybrid-positive contrastive learning, (2) injects diagnostic priors via
disease-aware cross-modal representation learning, and (3) lever-
ages radiologists’ gaze as probabilistic priors to guide attention to-
ward diagnostically salient regions. Extensive experiments demon-
strate that CoGaze consistently outperforms state-of-the-art meth-
ods across diverse tasks, achieving up to +2.0% CheXbertF1 and
+1.2% BLEU2 for free-text and structured report generation, +23.2%
AUROC for zero-shot classification, and +12.2% Precision@1 for
image-text retrieval. Code is available at https://github.com/mk-
runner/CoGaze.
Keywords
Medical vision-language pretraining, chest X-ray analysis, context-
and gaze-guided representation learning, report generation
1

---
## CXPMRG_Bench_Pretraining_Benchmarking_Xray_CVPR2025.pdf
**Abstract/Intro Start:**
X-ray image-based medical report generation (MRG) is
a pivotal area in artificial intelligence which can signifi-
cantly reduce diagnostic burdens and patient wait times.
Despite significant progress, we believe that the task has
reached a bottleneck due to the limited benchmark datasets
and the existing large models’ insufficient capability en-
hancements in this specialized domain.
Specifically, the
recently released CheXpert Plus dataset lacks compara-
tive evaluation algorithms and their results, providing only
the dataset itself. This situation makes the training, eval-
uation, and comparison of subsequent algorithms chal-
lenging. Thus, we conduct a comprehensive benchmark-
ing of existing mainstream X-ray report generation models
and large language models (LLMs), on the CheXpert Plus
dataset. We believe that the proposed benchmark can pro-
vide a solid comparative basis for subsequent algorithms
and serve as a guide for researchers to quickly grasp the
state-of-the-art models in this field. More importantly, we
propose a large model for the X-ray image report gener-
ation using a multi-stage pre-training strategy, including
self-supervised autoregressive generation and Xray-report
contrastive learning, and supervised fine-tuning. Extensive
experimental results indicate that the autoregressive pre-
training based on Mamba effectively encodes X-ray images,
and the image-text contrastive pre-training further aligns
the feature spaces, achieving better experimental results.
Source code can be found on https://github.com/
Event-AHU/Medical_Image_Analysis.
*  Corresponding Author: Bo Jiang

---
## Enhanced_Contrastive_Learning_MultiView_Longitudinal_CXR_CVPR2025.pdf
**Abstract/Intro Start:**
Automated radiology report generation offers an effective
solution to alleviate radiologists’ workload. However, most
existing methods focus primarily on single or fixed-view im-
ages to model current disease conditions, which limits di-
agnostic accuracy and overlooks disease progression. Al-
though some approaches utilize longitudinal data to track
disease progression, they still rely on single images to ana-
lyze current visits. To address these issues, we propose en-
hanced contrastive learning with Multi-view Longitudinal
data to facilitate chest X-ray Report Generation, named
MLRG. Specifically, we introduce a multi-view longitudinal
contrastive learning method that integrates spatial informa-
tion from current multi-view images and temporal informa-
tion from longitudinal data. This method also utilizes the
inherent spatiotemporal information of radiology reports to
supervise the pre-training of visual and textual represen-
tations. Subsequently, we present a tokenized absence en-
coding technique to flexibly handle missing patient-specific
prior knowledge, allowing the model to produce more accu-
rate radiology reports based on available prior knowledge.
Extensive experiments on MIMIC-CXR, MIMIC-ABN, and
Two-view CXR datasets demonstrate that our MLRG out-
performs recent state-of-the-art methods, achieving a 2.3%
BLEU-4 improvement on MIMIC-CXR, a 5.5% F1 score im-
provement on MIMIC-ABN, and a 2.7% F1 RadGraph im-
provement on Two-view CXR.

---
## FactCheXcker_Mitigating_Measurement_Hallucinations_CXR_CVPR2025.pdf
**Abstract/Intro Start:**
Medical vision-language models often struggle with gener-
ating accurate quantitative measurements in radiology re-
ports, leading to hallucinations that undermine clinical re-
liability.
We introduce FactCheXcker, a modular frame-
work that de-hallucinates radiology report measurements
by leveraging an improved query-code-update paradigm.
Specifically, FactCheXcker employs specialized modules
and the code generation capabilities of large language mod-
els to solve measurement queries generated based on the
original report. After extracting measurable findings, the
results are incorporated into an updated report. We eval-
uate FactCheXcker on endotracheal tube placement, which
accounts for an average of 78% of report measurements,
using the MIMIC-CXR dataset and 11 medical report-
generation models. Our results show that FactCheXcker
significantly reduces hallucinations, improves measurement
precision, and maintains the quality of the original reports.
Specifically, FactCheXcker improves the performance of
10/11 models and achieves an average improvement of
135.0% in reducing measurement hallucinations measured
by mean absolute error.
Code is available at https:
//github.com/rajpurkarlab/FactCheXcker.

---
## VILA-M3_Enhancing_VLMs_with_Medical_Expert_Knowledge_CVPR2025.pdf
**Abstract/Intro Start:**
:
Generalist vision language models (VLMs) have made significant strides in computer vision, but
they fall short in specialized fields like healthcare, where expert knowledge is essential. Current large multimodal
models like Gemini and GPT-4o are insufficient for medical tasks due to their reliance on memorized internet
knowledge rather than the nuanced expertise required in healthcare. Meanwhile, existing medical VLMs (e.g.
Med-Gemini) often lack expert consultation as part of their design, and many rely on outdated, static datasets
that were not created with modern, large deep learning models in mind. VLMs are usually trained in three stages:
vision pre-training, vision-language pre-training, and instruction fine-tuning (IFT). IFT has been typically
applied using a mixture of generic and healthcare data. In contrast, we propose that for medical VLMs, a
fourth stage of specialized IFT is necessary, which focuses on medical data and includes information from
domain expert models. Domain expert models developed for medical use are crucial because they are specifically
trained for certain clinical tasks, e.g. to detect tumors and classify abnormalities through segmentation and
classification, which learn fine-grained features of medical data−features that are often too intricate for a VLM
to capture effectively. This paper introduces a new framework, VILA-M3, for medical VLMs that utilizes domain
knowledge via expert models. We argue that generic VLM architectures alone are not viable for real-world
clinical applications and on-demand usage of domain-specialized expert model knowledge is critical for advancing
AI in healthcare. Through our experiments, we show an improved state-of-the-art (SOTA) performance with an
average improvement of ∼9% over the prior SOTA model Med-Gemini and ∼6% over models trained on the
specific tasks. Our approach emphasizes the importance of domain expertise in creating precise, reliable VLMs
for medical applications.
Links:
Code (GitHub) | Models (Hugging Face) | Demo

---
## X-WIN_Chest_Radiograph_World_Model_CVPR2026.pdf
**Abstract/Intro Start:**
Chest X-ray radiography (CXR) is an essential medical
imaging technique for disease diagnosis. However, as 2D
projectional images, CXRs are limited by structural super-
position and hence fail to capture 3D anatomies. This limi-
tation makes representation learning and disease diagnosis
challenging. To address this challenge, we propose a novel
CXR world model named X-WIN, which distills volumetric
knowledge from chest computed tomography (CT) by learn-
ing to predict its 2D projections in latent space. The core
idea is that a world model with internalized knowledge of
3D anatomical structure can predict CXRs under various
transformations in 3D space. During projection prediction,
we introduce an affinity-guided contrastive alignment loss
that leverages mutual similarities to capture rich, corre-
lated information across projections from the same volume.
To improve model adaptability, we incorporate real CXRs
into training through masked image modeling and employ
a domain classifier to encourage statistically similar rep-
resentations for real and simulated CXRs. Comprehensive
experiments show that X-WIN outperforms existing founda-
tion models on diverse downstream tasks using linear prob-
ing and few-shot fine-tuning. X-WIN also demonstrates the
ability to render 2D projections for reconstructing a 3D CT
volume.

---
## HKRG_Hierarchical_Knowledge_Integration_Radiology_ESWA2025.pdf
**Abstract/Intro Start:**
Current research on trajectory prediction primarily relies on data collected by onboard sensors of
an ego vehicle. With the rapid advancement in connected technologies, such as vehicle-to-vehicle
(V2V) and vehicle-to-infrastructure (V2I) communication, valuable information from alternate views
becomes accessible via wireless networks. The integration of information from alternative views
has the potential to overcome the inherent limitations associated with a single viewpoint, such
as occlusions and limited field of view. In this work, we introduce V2INet, a novel trajectory
prediction framework designed to model multi-view data by extending existing single-view models.
Unlike previous approaches where the multi-view data is manually fused or formulated as a separate
training stage, our model supports end-to-end training, enhancing both flexibility and performance.
Moreover, the predicted multimodal trajectories are calibrated by a post hoc conformal prediction
module to get valid and efficient confidence regions. We evaluated the entire framework on the
real-world V2I dataset V2X-Seq. Our results demonstrate superior performance in terms of Final
Displacement Error (FDE) and Miss Rate (MR) using a single GPU. The code is publicly available at:
https://github.com/xichennn/V2I_trajectory_prediction.
1

---
## GEMeX_Large_Scale_Groundable_Explainable_Medical_VQA_ICCV2025.pdf
**Abstract/Intro Start:**
Medical Visual Question Answering (Med-VQA) combines
computer vision and natural language processing to auto-
matically answer clinical inquiries about medical images.
However, current Med-VQA datasets exhibit two signifi-
cant limitations: (1) they often lack visual and textual ex-
planations for answers, hindering comprehension for pa-
tients and junior doctors; (2) they typically offer a narrow
range of question formats, inadequately reflecting the di-
verse requirements in practical scenarios.
These limita-
tions pose significant challenges to the development of a re-
liable and user-friendly Med-VQA system. To address these
challenges, we introduce a large-scale, Groundable, and
Explainable Medical VQA benchmark for chest X-ray diag-
nosis (GEMeX), featuring several innovative components:
(1) a multi-modal explainability mechanism that offers de-
tailed visual and textual explanations for each question-
answer pair, thereby enhancing answer comprehensibil-
ity; (2) four question types—open-ended, closed-ended,
single-choice, and multiple-choice—to better reflect prac-
tical needs. With 151,025 images and 1,605,575 questions,
GEMeX is the currently largest chest X-ray VQA dataset.
Evaluation of 12 representative large vision language mod-
els (LVLMs) on GEMeX reveals suboptimal performance,
underscoring the dataset’s complexity. Meanwhile, we pro-
pose a strong model by fine-tuning an existing LVLM on the
GEMeX training set. The substantial performance improve-
ment showcases the dataset’s effectiveness. The benchmark
is available at www.med-vqa.com/GEMeX.

---
## BioBridge_Bridging_Biomedical_Foundation_Models_ICLR2024.pdf
**Abstract/Intro Start:**
Foundation models (FMs) learn from large volumes of unlabeled data to demon-
strate superior performance across a wide range of tasks. However, FMs de-
veloped for biomedical domains have largely remained unimodal, i.e., indepen-
dently trained and used for tasks on protein sequences alone, small molecule
structures alone, or clinical data alone. To overcome this limitation, we present
BioBRIDGE, a parameter-efficient learning framework, to bridge independently
trained unimodal FMs to establish multimodal behavior. BioBRIDGE achieves it
by utilizing Knowledge Graphs (KG) to learn transformations between one uni-
modal FM and another without fine-tuning any underlying unimodal FMs. Our
results demonstrate that BioBRIDGE can beat the best baseline KG embedding
methods (on average by ∼76.3%) in cross-modal retrieval tasks. We also identify
BioBRIDGE demonstrates out-of-domain generalization ability by extrapolating
to unseen modalities or relations. Additionally, we also show that BioBRIDGE
presents itself as a general-purpose retriever that can aid biomedical multimodal
question answering as well as enhance the guided generation of novel drugs. 1
1

---
## MedRegA_Interpretable_Bilingual_Multimodal_LLM_ICLR2025.pdf
**Abstract/Intro Start:**
Several medical Multimodal Large Languange Models (MLLMs) have been de-
veloped to address tasks involving visual images with textual instructions across
various medical modalities, achieving impressive results. Most current medical
generalist models are region-agnostic, treating the entire image as a holistic rep-
resentation. However, they struggle to identify which specific regions they are
focusing on when generating a sentence. To mimic the behavior of doctors, who
typically begin by reviewing the entire image before concentrating on specific
regions for a thorough evaluation, we aim to enhance the capability of medi-
cal MLLMs in understanding anatomical regions within entire medical scans.
To achieve it, we first formulate Region-Centric tasks and construct a large-
scale dataset, MedRegInstruct, to incorporate regional information into train-
ing. Combining our collected dataset with other medical multimodal corpora for
training, we propose a Region-Aware medical MLLM, MedRegA, which is the
first bilingual generalist medical AI system to simultaneously handle image-level
and region-level medical vision-language tasks across a broad range of modali-
ties. Our MedRegA not only enables three region-centric tasks, but also achieves
the best performance for visual question answering, report generation and medi-
cal image classification over 8 modalities, showcasing significant versatility. Ex-
periments demonstrate that our model can not only accomplish powerful per-
formance across various medical vision-language tasks in bilingual settings, but
also recognize and detect structures in multimodal medical scans, boosting the
interpretability and user interactivity of medical MLLMs. Our project page is
https://medrega.github.io.
Figure 1: MedRegA, an interpretable bilingual generalist model for diverse biomedical tasks, repre-
sented by its outstanding ability to leverage regional information. MedRegA can perceive 8 modal-
ities covering almost all the body parts, showcasing significant versatility.
∗Corresponding to Xiaomeng Li (eexmli@ust.hk). 1The Hong Kong University of Science and Technology.
2Sun Yat-Sen Memorial Hospital, Sun Yat-Sen University.
1
arXiv:2410.18387v4  [cs.CV]  7 Apr 2025

Published as a conference paper at ICLR 2025
Figure 2: The significance of Region-Centric ability. (a) Comparison between the region-agnostic
model (MedDr) and the region-centric MedRegA in analyzing lesion area within the medical scan.
(b) Performance comparison of prompting the model with and without regional information on five
benchmarks of Visual Question Answering (VQA) and classification tasks.
1

---
## Learning_Self-Critiquing_Mechanisms_Region_Guided_CXR_ICLR2026.pdf
**Abstract/Intro Start:**
Clinically accurate and interpretable automatic radiology reporting requires re-
liably grounding the identified abnormalities with the corresponding regions lo-
cated in the radiology image. In this paper, we propose to introduce self-critiquing
mechanisms into the automatic report generation process to ensure the identified
abnormalities can reliably grounded before they are reported. Instead of adopting
LLM-based reasoning to implement the self-critiquing mechanisms which will in-
cur high inference cost in test time, we propose a novel Radiology Self-Critiquing
Reporting (RadSCR) model framework which allows multi-faceted mechanisms
to be learned end-to-end to identify and verify some hypothesized abnormality
regions by comparing with i) alternative abnormalities, ii) alternative patients’
X-ray images, and iii) potential false negatives. The self-critiqued abnormality
proposals are then integrated using a retrieval-based approach to generate the final
report. Our experimental results show that RadSCR can outperform the state-of-
the-art report generation methods in terms of clinical accuracy by a large margin,
with improved reliability of abnormality localization.
1

---
## Rethinking_Radiology_Report_Generation_Topic_Guided_ICLR2026.pdf
**Abstract/Intro Start:**
Vision-Language Models (VLMs) for radiology report generation are typically
trained to mimic the narrative flow of human experts. However, we identify a po-
tential limitation in this conventional paradigm. We hypothesize that optimizing
for narrative coherence encourages models to rely on linguistic priors and inter-
sentence correlations, which can weaken their grounding in direct visual evidence
and lead to factual inaccuracies. To investigate this, we design a controlled ex-
periment demonstrating that as textual context increases, a model’s reliance on
the input image systematically decays. We propose LLaVA-TA (Topic-guided
and Anatomy-aware), a new fine-tuning framework that directly addresses this
challenge by re-engineering the generation process. Instead of producing a linear
narrative, LLaVA-TA decomposes the report into a set of independent, clinically-
relevant topics. By training the model to generate a discrete finding for each topic
conditioned on both the full image and its corresponding anatomical region, we re-
duce the model’s reliance on narrative flow and enforce stricter visual grounding.
Our experiments show that LLaVA-TA sets a new state of the art on the MIMIC-
CXR dataset, significantly improving clinical accuracy on metrics like RadGraph
F1 (from 29.4 to 44.0) and CheXpert F1-14 (from 39.5 to 71.5) over strong base-
lines. Our work demonstrates that dismantling a report’s narrative structure to en-
force independent, visually-grounded observations is a crucial and effective step
toward building more accurate and reliable medical VLMs.
1

---
## MedRAX_Medical_Reasoning_Agent_Chest_Xray_ICML2025.pdf
**Abstract/Intro Start:**
Chest X-rays (CXRs) play an integral role in driv-
ing critical decisions in disease management and
patient care. While recent innovations have led
to specialized models for various CXR interpre-
tation tasks, these solutions often operate in iso-
lation, limiting their practical utility in clinical
practice. We present MedRAX, the first versatile
AI agent that seamlessly integrates state-of-the-
art CXR analysis tools and multimodal large lan-
guage models into a unified framework. MedRAX
dynamically leverages these models to address
complex medical queries without requiring addi-
tional training. To rigorously evaluate its capa-
bilities, we introduce ChestAgentBench, a com-
prehensive benchmark containing 2,500 complex
medical queries across 7 diverse categories. Our
experiments demonstrate that MedRAX achieves
state-of-the-art performance compared to both
open-source and proprietary models, representing
a significant step toward the practical deployment
of automated CXR interpretation systems. Data
and code have been publicly available at https:
//github.com/bowang-lab/MedRAX.

---
## Cyclic_Vision_Language_Manipulator_Radiology_IJCAI2025.pdf
**Abstract/Intro Start:**
Despite significant advancements in automated re-
port generation, the opaqueness of text inter-
pretability continues to cast doubt on the reliability
of the content produced. This paper introduces a
novel approach to identify specific image features
in X-ray images that influence the outputs of re-
port generation models. Specifically, we propose
Cyclic Vision-Language Manipulator (CVLM), a
module to generate a manipulated X-ray from an
original X-ray and its report from a designated re-
port generator. The essence of CVLM is that cy-
cling manipulated X-rays to the report generator
produces altered reports aligned with the alterations
pre-injected into the reports for X-ray generation,
achieving the term “cyclic manipulation”. This pro-
cess allows direct comparison between original and
manipulated X-rays, clarifying the critical image
features driving changes in reports and enabling
model users to assess the reliability of the gener-
ated texts. Empirical evaluations demonstrate that
CVLM can identify more precise and reliable fea-
tures compared to existing explanation methods,
significantly enhancing the transparency and appli-
cability of AI-generated reports.
1

---
## RRG-Mamba_Efficient_Radiology_Report_SSM_IJCAI2025.pdf
**Abstract/Intro Start:**
Recent advancements in radiology report genera-
tion have utilized deep neural networks such as
CNNs and Transformers, achieving notable im-
provements in generating accurate and detailed re-
ports.
However, their practical adoption is hin-
dered by the challenge of balancing global depen-
dency modeling with computational efficiency. The
state space model, particularly its enhanced variant
Mamba, offers promising linear-complexity solu-
tions for long-range dependency modeling. Despite
its strengths, Mamba’s fixed positional encoding
limits its ability to effectively capture complex spa-
tial dependencies. To address this gap, we propose
RRG-Mamba, an advanced framework for efficient
radiology report generation.
Within the RRG-
Mamba, we enhance the vanilla Mamba by inte-
grating rotary position encoding (RoPE), enabling
dynamic modeling of relative positional informa-
tion in visual feature sequences. Furthermore, we
design a global dependency learning module to op-
timize long-range visual feature sequence model-
ing. Extensive experiments on publicly available
datasets, including IU X-Ray and MIMIC-CXR,
demonstrate that RRG-Mamba achieves a 3.7% im-
provement in BLEU-4 score over existing mod-
els, along with significant gains in computational
and memory efficiency. Our code is available at
https://github.com/Eleanorhxd/RRG-Mamba.
1

---
## R2Gen-Mamba_Selective_State_Space_Model_Radiology_ISBI2025.pdf
**Abstract/Intro Start:**
Radiology report generation is crucial in medical imaging,
but the manual annotation process by physicians is time-
consuming and labor-intensive, necessitating the develop-
ment of automatic report generation methods.
Existing
research predominantly utilizes Transformers to generate
radiology reports, which can be computationally intensive,
limiting their use in real applications. In this work, we present
R2Gen-Mamba, a novel automatic radiology report genera-
tion method that leverages the efficient sequence processing
of the Mamba with the contextual benefits of Transformer
architectures.
Due to lower computational complexity of
Mamba, R2Gen-Mamba not only enhances training and in-
ference efficiency but also produces high-quality reports.
Experimental results on two benchmark datasets with more
than 210,000 X-ray image-report pairs demonstrate the ef-
fectiveness of R2Gen-Mamba regarding report quality and
computational efficiency compared with several state-of-the-
art methods. The source code can be accessed online.
Index Terms— Radiology, Report Generation, Selective
Satte Space Model, Transformer, Mamba

---
## CXRL_Text_Driven_CXR_Generation_Reinforcement_Learning_MICCAI2024.pdf
**Abstract/Intro Start:**
. Recent advances in text-conditioned image generation dif-
fusion models have begun paving the way for new opportunities in the
modern medical domain, in particular, particularly in generating Chest
X-rays (CXRs) from diagnostic reports. Nonetheless, to further drive the
diffusion models to generate CXRs that faithfully reflect the complexity
and diversity of real data, it has become evident that a nontrivial learn-
ing approach is needed. In light of this, we propose CXRL, a framework
motivated by the potential of reinforcement learning (RL). Specifically,
we integrate a policy gradient RL approach with well-designed multiple
distinctive CXR-domain specific reward models. This approach guides
the diffusion denoising trajectory, achieving precise CXR posture and
pathological details. Here, considering the complex medical image envi-
ronment, we present “RL with Comparative Feedback” (RLCF) for the
reward mechanism, a human-like comparative evaluation that is known
to be more effective and reliable in complex scenarios compared to direct
evaluation. Our CXRL framework includes jointly optimizing learnable
adaptive condition embeddings (ACE) and the image generator, enabling
the model to produce more accurate and higher perceptual CXR qual-
ity. Our extensive evaluation of the MIMIC-CXR-JPG dataset demon-
strates the effectiveness of our RL-based tuning approach. Consequently,
our CXRL generates pathologically realistic CXRs, establishing a new
standard for generating CXRs with high fidelity to real-world clinical
scenarios. Project page: https://micv-yonsei.github.io/cxrl2024/
Keywords: Chest X-Ray · Diffusion Models · Reinforcement Learning
1

---
## ECRG_Energy_Based_Controllable_Radiology_Report_MICCAI2024.pdf
**Abstract/Intro Start:**
. Automated generation of radiology reports from chest X-rays
has the potential to substantially reduce the workload of radiologists.
Recent advances in report generation using deep learning algorithms
have achieved significant results, benefiting from the incorporation of
medical knowledge. However, incorporation of additional knowledge or
constraints in existing models often require either altering network struc-
tures or task-specific fine-tuning. In this paper, we propose an energy-
based controllable report generation method, named ECRG. Specifically,
our method directly utilizes diverse off-the-shelf medical expert models
or knowledge to design energy functions, which are integrated into pre-
trained report generation models during the inference stage, without any
alterations to the network structure or fine-tuning. We also propose an
acceleration algorithm to improve the efficiency of sampling the complex
multi-modal distribution of report generation. ECRG is model-agnostic
and can be readily used for other pre-trained report generation models.
Two cases are presented on the design of energy functions tailored to
medical expert systems and knowledge. The experiments on widely used
datasets Chest ImaGenome v1.0.0 and MIMIC-CXR demonstrate the
effectiveness of our proposed approach.
Keywords: Radiology report generation · Chest X-ray · Energy based
model · Controllable generation
1

---
## SEI_Structural_Entities_Extraction_Patient_Indications_MICCAI2024.pdf
**Abstract/Intro Start:**
. The automated generation of imaging reports proves invalu-
able in alleviating the workload of radiologists. A clinically applicable
reports generation algorithm should demonstrate its effectiveness in pro-
ducing reports that accurately describe radiology findings and attend to
patient-specific indications. In this paper, we introduce a novel method,
Structural Entities extraction and patient indications Incorporation (SEI)
for chest X-ray report generation. Specifically, we employ a structural
entities extraction (SEE) approach to eliminate presentation-style vo-
cabulary in reports and improve the quality of factual entity sequences.
This reduces the noise in the following cross-modal alignment module by
aligning X-ray images with factual entity sequences in reports, thereby
enhancing the precision of cross-modal alignment and further aiding the
model in gradient-free retrieval of similar historical cases. Subsequently,
we propose a cross-modal fusion network to integrate information from
X-ray images, similar historical cases, and patient-specific indications.
This process allows the text decoder to attend to discriminative fea-
tures of X-ray images, assimilate historical diagnostic information from
similar cases, and understand the examination intention of patients.
This, in turn, assists in triggering the text decoder to produce high-
quality reports. Experiments conducted on MIMIC-CXR validate the
superiority of SEI over state-of-the-art approaches on both natural lan-
guage generation and clinical efficacy metrics. The code is available at
https://github.com/mk-runner/SEI.
Keywords: Chest X-ray report generation · Structural entities extrac-
tion · Patient-specific indications · Cross-modal fusion · Similar historical
cases.

2
K. Liu et al.
1

---
## CURV_Coherent_Uncertainty_Aware_Reasoning_VLMs_CXR_NeurIPS2025.pdf
**Abstract/Intro Start:**
Recent advancements in multimodal models have significantly improved vision-
language (VL) alignment in radiology. However, existing approaches struggle to
effectively utilize complex radiology reports for learning and offer limited inter-
pretability through attention probability visualizations. To address these challenges,
we introduce RadZero, a novel framework for VL alignment in chest X-ray with
zero-shot multi-task capability. A key component of our approach is VL-CABS
(Vision-Language Cross-Attention Based on Similarity), which aligns text em-
beddings with local image features for interpretable, fine-grained VL reasoning.
RadZero leverages large language models to extract concise semantic sentences
from radiology reports and employs multi-positive contrastive training to effectively
capture relationships between images and multiple relevant textual descriptions.
It uses a pre-trained vision encoder with additional trainable Transformer layers,
allowing efficient high-resolution image processing. By computing similarity be-
tween text embeddings and local image patch features, VL-CABS enables zero-shot
inference with similarity probability for classification, and pixel-level VL similar-
ity maps for grounding and segmentation. Experimental results on public chest
radiograph benchmarks show that RadZero outperforms state-of-the-art methods in
zero-shot classification, grounding, and segmentation. Furthermore, VL similarity
map analysis highlights the potential of VL-CABS for improving explainability in
VL alignment. Additionally, qualitative evaluation demonstrates RadZero’s capabil-
ity for open-vocabulary semantic segmentation, further validating its effectiveness
in medical imaging. Code is available at https://github.com/deepnoid-ai/RadZero.
1

---
## R2GenGPT_Radiology_Report_Generation_with_Frozen_LLMs.pdf
**Abstract/Intro Start:**
Large Language Models (LLMs) have consistently showcased remarkable generalization capa-
bilities when applied to various language tasks. Nonetheless, harnessing the full potential of
LLMs for Radiology Report Generation (R2Gen) still presents a challenge, stemming from the
inherent disparity in modality between LLMs and the R2Gen task. To bridge this gap effectively,
we propose R2GenGPT, which is a novel solution that aligns visual features with the word
embedding space of LLMs using an efficient visual alignment module. This innovative approach
empowers the previously static LLM to seamlessly integrate and process image information,
marking a step forward in optimizing R2Gen performance. R2GenGPT offers the following
benefits. First, it attains state-of-the-art (SOTA) performance by training only the lightweight
visual alignment module while freezing all the parameters of LLM. Second, it exhibits high
training efficiency, as it requires the training of an exceptionally minimal number of parameters
while achieving rapid convergence. By employing delta tuning, our model only trains 5M
parameters (which constitute just 0.07% of the total parameter count) to achieve performance
close to the SOTA levels. Our code is available at https://github.com/wang-zhanyu/R2GenGPT.

---
## CMCRL_CrossModal_Causal_Representation_Learning_TIP2025.pdf
**Abstract/Intro Start:**
—Medical phrase grounding is crucial for identifying
relevant regions in medical images based on phrase queries,
facilitating accurate image analysis and diagnosis. However,
current methods rely on manual extraction of key phrases from
medical reports, reducing efficiency and increasing the workload
for clinicians. Additionally, the lack of model confidence estimation
limits clinical trust and usability. In this paper, we introduce a
novel task called Medical Report Grounding (MRG), which aims
to directly identify diagnostic phrases and their corresponding
grounding boxes from medical reports in an end-to-end manner.
To address this challenge, we propose uMedGround, a robust and
reliable framework that leverages a multimodal large language
model to predict diagnostic phrases by embedding a unique token,
<BOX>, into the vocabulary to enhance detection capabilities. A
vision encoder-decoder processes the embedded token and input
image to generate grounding boxes. Critically, uMedGround
incorporates an uncertainty-aware prediction model, significantly
improving the robustness and reliability of grounding predictions.
Experimental results demonstrate that uMedGround outperforms
state-of-the-art medical phrase grounding methods and fine-tuned
large visual-language models, validating its effectiveness and
reliability. This study represents a pioneering exploration of
the MRG task, marking the first-ever endeavor in this domain.
Additionally, we demonstrate the applicability of uMedGround
in medical visual question answering and class-based localization
tasks, where it highlights visual evidence aligned with key
Manuscript received December, 2024; revised June, 2025. This work was
supported by the National Key Researchand Development Program of China
under Grant 2020YFA0714003, the National Natural Science Foundation
of China (No.62025604, 62411540034), the H. Fu’s Agency for Science,
Technology and Research (A*STAR) Central Research Fund (“Robust and
Trustworthy AI system for Multi-modality Healthcare”), the Science and
Technology Department of Sichuan Province (Grant No. 2022YFS0071), the
Science and Technology Department of Guangxi Zhuang Autonomous Region
(Grant No. 2025GXNSFAA069531), Science and Technology Department
of Hainan Province (Grant number ZDYF2024SHFZ052) and the China
Scholarship Council (No. 202206240082).
K. Zou and X. Yuan are with the College of Computer Science, Sichuan
University, Chengdu, China.
Y. Bai, Y. Zhou, M. Wang, and H. Fu are with the Institute of High
Performance Computing (IHPC), Agency for Science, Technology and Research
(A*STAR), Singapore.
Bo Liu is with the Department of Computing, The Hong Kong Polytechnic
University, Hong Kong, China.
Y. Chen is with the Department of Radiology, West China Hospital, Sichuan
University, Chengdu, China.
Z. Chen is with the College of Intelligence and Computing, Tianjin
University, Tianjin, China
X. Shen is with the Department of Mathematics, Sichuan University,
Chengdu, China.
X. Cao is with the School of Cyber Science and Technology, Shenzhen
Campus of Sun Yat-Sen University, Shenzhen, China.
K. Zou and Y. Tham are with the Department of Ophthalmology, Yong Loo
Lin School of Medicine, National University of Singapore and the Singapore
Eye Research Institute, Singapore National Eye Centre, Singapore, Singapore.
Ke Zou and Yang Bai are equally contributed. Xuedong Yuan and
Huazhu Fu are the co-corresponding authors (e-mail: yxdongdong@163.com,
hzfu@ieee.org).
Fig. 1.
Medical report grounding. (a) Illustration of our medical report
grounding with large-language model for radiological diagnosis. (b) Different
paired cases (X-ray image, medical report, grounding box and key phrase):
Cardiomegaly and Pleural Effusion.
diagnostic phrases, supporting clinicians in interpreting various
types of textual inputs, including free-text reports, visual question
answering queries, and class labels.
Index Terms—Medical report grounding, vision-language model,
uncertainty estimation.
I.

---
## STREAM_SpatioTemporal_RetrievalAugmented_CXR_TMI2025.pdf
**Abstract/Intro Start:**
:
Creating high-quality controllable 3D human models from multi-view RGB videos poses a significant chal-
lenge. Neural radiance fields (NeRFs) have demonstrated remarkable quality in reconstructing and free-
viewpoint rendering of static as well as dynamic scenes. The extension to a controllable synthesis of dynamic
human performances poses an exciting research question. In this paper, we introduce a novel NeRF-based
framework for pose-dependent rendering of human performances. In our approach, the radiance field is warped
around an SMPL body mesh, thereby creating a new surface-aligned representation. Our representation can be
animated through skeletal joint parameters that are provided to the NeRF in addition to the viewpoint for pose
dependent appearances. To achieve this, our representation includes the corresponding 2D UV coordinates
on the mesh texture map and the distance between the query point and the mesh. To enable efficient learning
despite mapping ambiguities and random visual variations, we introduce a novel remapping process that re-
fines the mapped coordinates. Experiments demonstrate that our approach results in high-quality renderings
for novel-view and novel-pose synthesis.
Figure 1: Synthesized images of the hand wave sequence in novel view and novel pose.
arXiv:2311.03140v1  [cs.CV]  6 Nov 2023

1

---
