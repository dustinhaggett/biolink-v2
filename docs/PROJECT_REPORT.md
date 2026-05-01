# BioLink: AI-Powered Drug Repurposing Discovery

**AAI 595 — Applied Machine Learning**
**Dr. Tao Han, Drexel University**

**Team:** Dustin Haggett, Kera Prosper, Esume

---

## 1. Problem Statement

### 1.1 The Problem

Drug development is one of the most expensive and time-consuming processes in modern science. Bringing a new drug from discovery to market takes an average of 10-15 years and costs over $2.6 billion (DiMasi et al., 2016). The failure rate exceeds 90%. Meanwhile, thousands of FDA-approved drugs sit on shelves with well-understood safety profiles — any of which might treat conditions they were never designed for.

**Drug repurposing** — finding new therapeutic uses for existing drugs — offers a faster, cheaper path. Because these drugs already have established safety data, repurposed candidates can bypass years of preclinical testing. Notable successes include:

- **Thalidomide**: Originally a sedative, now treats multiple myeloma
- **Sildenafil (Viagra)**: Developed for angina, repurposed for erectile dysfunction
- **Metformin**: A diabetes drug showing promise in cancer and aging research

However, identifying repurposing candidates currently requires domain expertise, literature review, and significant manual effort. There is no accessible tool that allows patients, researchers, or clinicians to quickly explore which existing drugs might treat a given condition — grounded in real evidence rather than speculation.

### 1.2 Why AI?

The relationships between drugs and diseases form a massive knowledge graph with over 7,000 drugs, 2,500 diseases, and hundreds of thousands of documented interactions in databases like the Comparative Toxicogenomics Database (CTD). A machine learning model can learn latent patterns in this graph that humans cannot feasibly identify manually.

Our prior work (BioLink v1) demonstrated that an MLP classifier trained on BioWordVec embeddings of drug-disease pairs achieves an AUC of 0.947 on held-out CTD data — meaning the model can reliably distinguish real drug-disease associations from random pairings. BioLink v2 transforms this research model into a consumer-facing application that makes these predictions accessible, interpretable, and grounded in evidence.

### 1.3 Societal Impact

Drug repurposing has outsized impact on:

- **Rare diseases**: 95% of rare diseases have no FDA-approved treatment. Repurposing existing drugs is often the only economically viable path.
- **Global health equity**: Repurposed generics are dramatically cheaper than novel drugs, expanding access in low-resource settings.
- **Pandemic response**: During COVID-19, repurposing screens identified dexamethasone as a life-saving treatment within months — far faster than novel drug development.
- **Patient empowerment**: Patients with unmet medical needs can use BioLink to identify potential candidates for discussion with their healthcare providers.

BioLink does not replace clinical judgment. It is a hypothesis-generation tool that surfaces candidates for further investigation, always with appropriate disclaimers.

---

## 2. Project Description

### 2.1 Solution Approach

BioLink v2 is a full-stack web application that wraps a trained MLP classifier in a multi-stage pipeline:

1. **Natural language input** — Users type a disease name or drug name in plain English
2. **Entity resolution** — An LLM (Claude Haiku) maps free-text input to standardized CTD entities, with fuzzy matching fallback
3. **Model scoring** — The MLP scores all 7,163 drugs (or 2,525 diseases in reverse mode) and ranks them by calibrated probability
4. **Enrichment** — Parallel async calls to PubMed, OpenFDA, and ClinicalTrials.gov add real-world evidence to each prediction
5. **Evidence search** — Perplexity Sonar searches published literature for mechanism-of-action, clinical studies, and trial data
6. **Presentation** — Results are displayed with confidence badges, verdict badges, evidence quality indicators, and pathway chains

### 2.2 AI Model Details

**Architecture:** MLP Classifier (Multi-Layer Perceptron)
- Input: 800-dimensional feature vector (concatenation of drug embedding, disease embedding, absolute difference, element-wise product)
- Hidden layer: 256 neurons with BatchNorm, ReLU, and 30% dropout
- Output: Single logit (pre-sigmoid)
- Training data: ~380,000 drug-disease pairs from CTD
- Performance: AUC = 0.947 on held-out test set

**Embeddings:** BioWordVec (200-dimensional)
- Pre-trained on PubMed abstracts and MIMIC-III clinical notes
- Captures biomedical semantic relationships between terms
- Drug and disease names are tokenized, embedded per-token, and mean-pooled

**Calibration:** Temperature scaling
- Post-hoc calibration using a validation set to ensure predicted probabilities are well-calibrated
- Confidence tiers: Strong (>= 80%), Moderate (50-79%), Speculative (< 50%)

**Reverse Search:**
- The model architecture is symmetric — the feature vector construction works identically for scoring diseases against a fixed drug
- All 2,525 disease embeddings are pre-cached at startup for instant reverse scoring

### 2.3 Multi-API Integration

BioLink integrates five external APIs to ground predictions in real-world evidence:

| API | Purpose | Auth Required | Cost |
|-----|---------|---------------|------|
| Claude Haiku (Anthropic) | Entity resolution, explanations | API key | ~$0.001/query |
| Perplexity Sonar | Evidence search, follow-up Q&A | API key | ~$0.005/query |
| PubMed E-utilities (NCBI) | Publication co-occurrence counts | None | Free |
| OpenFDA | Drug approval status | None | Free |
| ClinicalTrials.gov v2 | Active/completed trial lookup | None | Free |

All enrichment calls run in parallel using Python's asyncio, adding minimal latency to the pipeline.

### 2.4 Key Features

**Evidence Verdict System:** Each drug-disease pair receives a verdict badge based on Perplexity's analysis of published literature:
- *Evidence Supports* (green) — Published evidence suggests therapeutic potential
- *Standard of Care* (blue) — Drug is already an established treatment
- *Evidence Conflicts* (red) — Evidence suggests the drug would be ineffective or harmful
- *Insufficient Evidence* (gray) — Not enough published data to draw conclusions

**Evidence Quality Indicator:** Classifies the strength of available evidence as RCT, Human Study, Preclinical, Case Report, or Theoretical.

**Pathway Visualization:** Shows the biological mechanism chain linking drug to disease (e.g., "Metformin -> AMPK activation -> mTOR inhibition -> anti-tumor effect").

**Drug Interaction Warnings:** Flags candidates that have known dangerous interactions with standard treatments for the queried disease.

**Clinical Trial Finder:** Automatically queries ClinicalTrials.gov for each candidate, showing active and completed trials with direct enrollment links.

**Compare Mode:** Users can select multiple candidates for side-by-side comparison across all metrics.

**Reverse Search:** "I'm taking Metformin — what else might it help?" Scores all 2,525 diseases against a single drug.

**Export:** CSV and PDF report generation for sharing results with healthcare providers.

### 2.5 Development Tools

| Tool | Usage |
|------|-------|
| Claude Code (Anthropic) | Architecture design, code generation, debugging, feature implementation, documentation |
| Claude API (Haiku) | Runtime entity resolution and explanation generation |
| Perplexity Sonar API | Runtime evidence search |
| GitHub Copilot | Code completion during early development |
| PyTorch | Model loading and inference |
| Streamlit | Web application framework |
| gensim | BioWordVec embedding loading |
| aiohttp | Async HTTP for parallel API calls |
| fpdf2 | PDF report generation |

### 2.6 Expected Impact

BioLink makes drug repurposing research accessible to three audiences:

1. **Patients** with unmet medical needs can explore potential candidates and bring evidence-backed suggestions to their physicians
2. **Researchers** can quickly screen candidates for further investigation, with direct links to clinical trials and published studies
3. **Clinicians** can compare candidates side-by-side, assess evidence quality, and identify drugs with known interactions

The tool explicitly disclaims medical authority and positions itself as a hypothesis generator — surfacing candidates that warrant further clinical investigation.

---

## 3. Architecture

### 3.1 System Architecture Diagram

```
                           User Input
                              |
                    [Natural Language Query]
                              |
                    +-------------------+
                    |  Intent Mapper    |
                    |  (Claude Haiku +  |
                    |   fuzzy fallback) |
                    +-------------------+
                              |
                    [CTD Entity (drug or disease)]
                              |
                    +-------------------+
                    |   MLP Classifier  |
                    |   (PyTorch)       |
                    |   + Temperature   |
                    |     Scaling       |
                    +-------------------+
                              |
                    [Ranked candidates with calibrated probabilities]
                              |
            +--------+--------+--------+---------+
            |        |        |        |         |
        +-------+ +------+ +------+ +-------+ +--------+
        |PubMed | |OpenFDA| | ClinT | |Perplx | |Claude  |
        |E-utils| |  API  | |.gov   | |Sonar  | |Explain |
        +-------+ +------+ +------+ +-------+ +--------+
            |        |        |        |         |
            +--------+--------+--------+---------+
                              |
                    [Enriched Results]
                              |
                    +-------------------+
                    |   Streamlit UI    |
                    |   - Result cards  |
                    |   - Verdict badges|
                    |   - Compare mode  |
                    |   - PDF/CSV export|
                    +-------------------+
                              |
                         User Output
```

### 3.2 Data Flow

1. User enters query (disease or drug name)
2. Intent mapper resolves to CTD entity via Claude Haiku (with difflib fuzzy fallback)
3. BioWordVec encodes the entity to a 200-dim vector
4. MLP scores all candidates in a single batched forward pass (~0.1s)
5. Temperature scaling converts logits to calibrated probabilities
6. Confidence tiers assigned (Strong/Moderate/Speculative)
7. Async enrichment gathers PubMed counts, FDA status, and clinical trials in parallel
8. Perplexity searches for evidence on top 5 candidates (verdict, TL;DR, pathway, quality, interactions)
9. Claude generates plain-English explanations for top 10 candidates
10. Results rendered in Streamlit with interactive cards, filters, and export options

### 3.3 File Structure

```
biolink-v2/
├── app.py                      # Streamlit entry point, session state, pipeline orchestration
├── core/
│   ├── model.py                # BioLinkModel: MLP + BioWordVec, score_all_drugs/diseases
│   ├── inference.py            # disease_to_drugs() and drug_to_diseases() pipelines
│   ├── intent_mapper.py        # map_disease() and map_drug() entity resolution
│   └── calibration.py          # TemperatureScaler, confidence_tier()
├── enrichment/
│   ├── runner.py               # Async orchestrator (gather PubMed + FDA + trials)
│   ├── pubmed.py               # PubMed E-utilities co-occurrence count
│   ├── openfda.py              # FDA approval status lookup
│   ├── clinicaltrials.py       # ClinicalTrials.gov v2 API trial finder
│   └── perplexity.py           # Evidence search, verdict/quality/pathway parsing
├── explanation/
│   └── explainer.py            # Claude Haiku plain-English explanations
├── ui/
│   ├── components.py           # Result cards, badges, filters, search, compare
│   ├── styles.py               # CSS injection (Manrope/Inter fonts, teal palette)
│   └── pdf_export.py           # PDF report generator (fpdf2)
├── data/
│   ├── drugs_list.txt          # 7,163 CTD drug names
│   ├── diseases_list.txt       # 2,525 CTD disease names
│   └── temperature.json        # Calibration parameter T
├── models/
│   └── biolink_v1.pt           # Trained MLP weights
├── tests/
│   ├── test_explainer.py       # Explanation generation tests
│   └── ...
├── requirements.txt
└── README.md
```

---

## 4. Challenges and Solutions

### Challenge 1: Model Confidence vs. Real-World Evidence

**Problem:** The MLP model assigns high confidence scores based on knowledge graph topology, but this doesn't always align with clinical evidence. For example, immunosuppressants like Cyclosporine scored 99% for Lyme Disease — a bacterial infection where immunosuppression would be harmful.

**Solution:** We integrated Perplexity Sonar to ground every prediction in published literature. The evidence verdict system (Supports/Conflicts/Standard-of-Care/Insufficient) gives users a second, independent signal. When the model says "99% Strong" but evidence says "Evidence Conflicts," users see both — and the conflict itself is informative.

**AI tools helped:** Claude Code designed the structured prompt that extracts verdict, TL;DR, evidence quality, pathway, and interaction data from a single Perplexity API call, minimizing cost while maximizing information density.

### Challenge 2: API Cost Management

**Problem:** With 20 drug candidates per query, calling Perplexity for each would cost ~$0.10/query. At scale, this becomes prohibitive for a student project.

**Solution:** We limit evidence search to the top 5 candidates (where it matters most), use Claude Haiku instead of Opus/Sonnet for entity resolution and explanations (~60x cheaper), and use free APIs (PubMed, OpenFDA, ClinicalTrials.gov) for basic enrichment on all 20 results.

**AI tools helped:** Claude Code identified the cost bottleneck and suggested the tiered approach — expensive evidence search for top 5, cheap enrichment for all 20, and Claude Haiku for intent mapping.

### Challenge 3: Entity Resolution for Free-Text Input

**Problem:** Users type diseases in many forms ("heart attack" vs "Myocardial Infarction" vs "MI"). The model needs exact CTD entity names to produce meaningful scores.

**Solution:** Two-tier entity resolution: Claude Haiku maps natural language to CTD entities with high accuracy, with difflib fuzzy matching as a fallback when the API is unavailable. The system prompt instructs the LLM to return confidence levels, and low-confidence matches trigger a clarification prompt.

**AI tools helped:** Claude Code designed the system prompt and fallback logic. It also identified a bug where Haiku returned "low" confidence for exact matches (unlike Opus), and fixed the intent mapper to skip clarification when the entity is valid.

### Challenge 4: Reverse Search Architecture

**Problem:** The original model only supported disease-to-drug search. Adding drug-to-disease required scoring all 2,525 diseases — but disease embeddings weren't pre-cached.

**Solution:** We pre-compute and cache all 2,525 disease embeddings at model load time (only 1.4MB extra memory). The `score_all_diseases()` method mirrors `score_all_drugs()` exactly — same feature vector construction, same batched forward pass, same calibration. This was architecturally feasible because the MLP's feature vector is symmetric.

**AI tools helped:** Claude Code analyzed the model architecture, confirmed the feature vector symmetry, and implemented `score_all_diseases()` as a mirror of the existing method with correct argument ordering.

### Challenge 5: PDF Export with Unicode

**Problem:** fpdf2's default fonts only support latin-1 encoding. Evidence text from Perplexity contains em dashes, smart quotes, and other Unicode characters that crash the PDF generator.

**Solution:** A `_safe()` function replaces common Unicode characters with ASCII equivalents and strips remaining non-latin-1 characters. A `_mc()` wrapper ensures the cursor is always at the left margin before `multi_cell` calls, preventing "not enough horizontal space" errors from cursor position drift.

**AI tools helped:** Claude Code identified the root cause (cursor position after auto page break) and iterated through three fixes before arriving at the robust solution.

---

## 5. Post-Presentation Findings and Improvements

After the in-class presentation, we returned to the deployed app with a question: which of the failure modes we documented during the demo (Cyclosporine for Lyme, Cocaine for Alcoholism, Scopolamine for Amnesia) reflect fixable system issues, and which reflect deeper limitations? The answers turned out to be paper-worthy in their own right.

### 5.1 The Honest AUC: A Popularity Baseline at 0.88

We reproduced v1's reported AUC of 0.947 on the original 50/50 pair-level test set (got 0.9497, exact match). But that test setup uses random negatives drawn from the chemical × disease cross-product — a trivial discrimination task compared to the real one. We then computed **per-disease ranking AUC** across all 2,526 diseases: for each disease, rank all 7,163 candidate drugs and compare against CTD's curated therapeutic indications.

| Method | AUC mean | AUC median | What it is |
|---|---|---|---|
| **Trained MLP** | **0.9470** | **0.9770** | The deployed model |
| **Drug popularity** | **0.8803** | **0.9083** | Rank drugs by global # therapeutic indications. Same ranking for every disease. |
| Cosine similarity | 0.8008 | 0.8420 | Plain BioWordVec cosine, no ML at all |
| Random | 0.5099 | 0.5072 | Sanity check |

**The MLP only adds +6.7 points (mean) above a popularity baseline that uses the same drug ranking for every disease.** ~93% of the model's apparent performance is captured by global drug popularity — the model has learned mostly "which drugs appear in many therapeutic pairs," not disease-specific therapeutic patterns.

This is consistent with the cluster-mismatch failures we observed during the demo. For 16.9% of diseases (typically rare conditions with 1–2 known therapeutics), popularity actively beats the MLP. AUC near 1.0 also masks user-facing failures: Lyme Disease shows MLP AUC = 0.999 (only 2 positives among 7,164 candidates makes ranking trivial), but the model's #1 ranked drug for Lyme was Cyclosporine — wrong and clinically harmful.

### 5.2 Hard-Negative Retraining: An Inverse-Popularity Trade-off

The most direct fix for the documented direction-blindness failures (Cocaine for Alcoholism, Scopolamine for Amnesia, Streptozocin for T2D) is to teach the model what "harmful" looks like by sampling hard negatives from CTD's `marker/mechanism` evidence rows — pairs where the drug is causally associated with the disease but is not therapeutic.

We retrained three variants using the same v1 architecture, hyperparameters, and seed:

| Variant | Val AUC | Wins preserved (of 6) | Failures improved (of 6) |
|---|---|---|---|
| Baseline (v1, random negatives) | 0.9497 | 6/6 | 0/6 |
| **100% hard negatives** | 0.9161 | **0/6** | 5/6 |
| 50/50 random + hard | 0.9005 | 3/6 | 5/6 |
| 25/75 random + hard | 0.9?? | 4/6 | 5/6 |

Pure hard-negative training produced an **inverse popularity bias**. Every disease's top 10 became dominated by obscure plant compounds and traditional Chinese medicine ingredients — *Corilagin* appeared at #1 for Migraine, Fibromyalgia, AND Amnesia. The cause: marker/mechanism is dominated by the *same heavily-studied common drugs* as the therapeutic class (cyclosporine, methotrexate, etc.). Training only on hard negatives taught the model "well-studied drugs = NEGATIVE," and obscure compounds with no marker/mechanism record floated to the top.

Mixed sampling (25/75 and 50/50) recovered most baseline wins while still demoting the most obvious harm cases, but **no mix is unambiguously better**. The trade-off is irreducible: more hard negatives → safer on direction-blindness → degraded ranking on documented wins. We chose not to ship a retrained model and instead built a complementary safety mechanism (Section 5.4).

### 5.3 Calibration Bug: Score Saturation and the Prior-Shift Fix

During the regression review, we noticed every result in the regression suite scored 97–100% confidence. The MLP was uniformly overconfident, making the displayed probabilities uninformative. Root cause: the model was trained on a 1:1 positive:negative class ratio but used at inference time for a fundamentally different task — out of 7,163 candidate drugs, the real prior of "this one treats this disease" is ~1%.

We applied **Bayesian prior correction**: subtract a constant from `logit/T` before the sigmoid, where the constant is derived from the train-vs-inference prior mismatch:

```python
shift = log_odds(p_train) - log_odds(p_real)
     = log(0.5/0.5) - log(0.01/0.99)
     ≈ 4.595
P(therapeutic | features, real_prior) = sigmoid(logit/T - shift)
```

The transform is monotonic, so ranking is preserved exactly. The displayed probabilities now reflect the realistic prior. The same regression suite went from **240/240 results showing as Strong** (uniformly saturated) to **197 Strong / 43 Moderate / 0 Speculative** — meaningful score spread is now visible to users.

We also adjusted the `confidence_tier` thresholds (Strong ≥ 0.30, Moderate ≥ 0.10) to match the new distribution. Original thresholds (0.80 / 0.50) were calibrated for the saturated pre-correction world.

A surprising downstream finding: per-disease tier counts now reveal cluster-mismatch failures visibly. Most diseases have 4–80 Strong candidates. **Amnesia produces 951 Strong candidates** (13% of the entire drug pool) — direct evidence that the model's embedding for Amnesia is in a problematic region of vector space.

### 5.4 Harm-Aware Reranking: From Scores to Safety

Rather than retrain (which traded one bias for another), we layered a **harm-aware reranker** on top of the deployed model. The reranker uses the Perplexity evidence-layer verdicts to adjust scores after inference, with a deliberate design constraint: **never demote on absence of evidence** — that's the discovery space the system exists to surface.

We added a new structured field to the Perplexity prompt — `HARM_FOR_INDICATION: HARMFUL/NOT_HARMFUL/UNKNOWN` — and asked Perplexity a clinical-practice question rather than a literature-evidence question:

> *"If a clinician administered {drug} to a patient WHO HAS {disease}, would the patient likely be harmed, or their condition worsened, based on the drug's known mechanism, pharmacology, or contraindications?"*

This phrasing was deliberate. An earlier version asked about published evidence directly, which scored 4/6 on a designed validation set. The patient-outcome framing scored **19/20** on an expanded 20-case validation set across 10 documented harm cases, 7 documented wins, and 3 emerging discovery candidates.

The reranker applies multipliers based on a five-rule precedence:

| Rule | Multiplier | Reason |
|---|---|---|
| 1. `HARM_FOR_INDICATION = HARMFUL` | × 0.3 (× 0.1 with interactions) | Indication-specific harm |
| 2. `verdict = STANDARD-OF-CARE` | × 1.5 | Established treatment |
| 2. `verdict = SUPPORTS` | × 1.3 | Evidence supports |
| 3. `has_interactions` AND no positive validation | × 0.5 | Backstop for missed harm classifications |
| 4. `verdict = INSUFFICIENT` AND active trials AND no interactions | × 1.15 | Discovery boost |
| 5. Default | × 1.0 | Preserve (CONFLICTS without harm, INSUFFICIENT without trials) |

**Critical:** Rule 5 (default × 1.0) is what makes the system a discovery tool rather than a clinical reference. Mixed-evidence and unstudied candidates are preserved at their model score. Only candidates with evidence of indication-specific harm get demoted.

In production deployment on the documented test cases:
- Cyclosporine for Lyme moved from **#1 (99% Strong) → #9 (20% Moderate)**
- Cocaine, Heroin, PCP, Methamphetamine for Alcoholism all dropped to **the bottom of the visible list (5–10% Speculative)** with HARMFUL or interactions badges
- Topiramate for Migraine remained at **#1 with × 1.5 boost (now visible at 100% Strong)**
- Naltrexone for Fibromyalgia jumped from **#6 → #6 with × 1.3 discovery boost** ("Evidence Supports" + active LDN trials)

We bumped Perplexity enrichment from `top_n=5` to `top_n=20` so every visible candidate is eligible for rerank — cost increases from ~$0.025/query to ~$0.10/query, but harm cases at ranks 6–20 (e.g., Strychnine #10 for Amnesia, Methylprednisolone for Lyme) are now demoted instead of slipping through.

### 5.5 The Strychnine Paradox: Limits of Literature-Driven Judges

End-to-end validation against 20 designed (drug, disease) pairs scored 19/20. The single non-match is illuminating:

> **Strychnine for Amnesia → NOT_HARMFUL** (we expected HARMFUL).
> Perplexity TLDR: *"Preclinical animal studies from the 1970s-1980s suggest strychnine may attenuate experimentally induced amnesia, but no human data or clinical trials exist for repurposing it as an amnesia treatment."*

Perplexity is being **narrowly correct**: at sub-toxic doses, there is old preclinical literature suggesting strychnine has memory-enhancing effects via glycine-receptor antagonism. But strychnine is a **convulsant poison** with no clinical safety margin. The system reasoned over the literature it found and missed the broader common-sense safety concern. We see the same pattern with Picrotoxin (preclinical Alzheimer's research showing memory enhancement at sub-toxic doses).

This is an honest limitation of literature-driven LLM judges. A clinical-deployment system would need a controlled-substance / poison blacklist as an additional safety layer — DEA Schedule I substances and known poisons should be filtered regardless of preclinical "memory benefit" papers. We did not implement this for the academic demo, but documenting the failure mode is a legitimate contribution.

---

## 6. Limitations and Future Work

### 6.1 The Discovery-vs-Harm Tension

Drug repurposing tools exist to surface unexpected uses. By definition, the most valuable predictions are candidates that don't yet have established evidence — that's the discovery space. A naïve safety filter ("only show validated treatments") would make the tool useless for its primary purpose.

Our harm-aware reranker is **deliberately asymmetric**: aggressive demotion when evidence of indication-specific harm exists, no demotion when evidence is merely insufficient. The Naltrexone-for-Fibromyalgia case is the discovery-preservation success — Perplexity returned `verdict=SUPPORTS` with active clinical trial activity, the discovery boost fired, and the candidate moved from rank #6 to a more visible position with an "Evidence Supports" badge.

### 6.2 What This System Cannot Do

We chose to be explicit about the system's limits rather than overclaim:

1. **No mechanistic reasoning.** The pipeline treats drugs and diseases as text, not biology. A drug whose target protein is upregulated in a disease cannot be matched to that disease unless they co-occur in PubMed text. First-degree novel discoveries (new mechanism → new indication) are out of scope.

2. **CTD discovery floor.** The training set's vocabulary is set by the Comparative Toxicogenomics Database, which heavily favors pharmaceutical-industry research. Traditional medicine, herbal compounds, food-derived molecules, and off-patent drugs are systematically underrepresented — exactly the space repurposing should explore most aggressively.

3. **Cluster-mismatch failures persist.** Even with prior correction and harm-aware reranking, the model still favors drugs popular in the closest embedding cluster. Lyme Disease is embedded near autoimmune diseases (because of shared symptom vocabulary), so immunosuppressants will continue to surface in the candidate pool. Hard-negative training reduced this somewhat but introduced an inverse popularity bias; a knowledge-graph embedding (TransE / ComplEx on DrugBank+CTD relations) would address it more cleanly but was beyond the scope of this work.

4. **Trial-outcome blindness.** The model rewards research interest regardless of trial outcomes. ALS produces a top-20 dominated by drugs that were trialed and failed (memantine, lithium, minocycline). Fixing this requires ClinicalTrials.gov outcome features as model inputs, not just retrieval.

5. **Literature-driven harm classification has a common-sense floor.** Section 5.5 documents the Strychnine paradox. Even with the patient-outcome prompt framing, an LLM judge can miss safety concerns that "everyone knows but nobody published."

6. **Trial misattribution.** ClinicalTrials.gov returns trials that mention the drug name, even when the drug is not the intervention. Methamphetamine searches returned MDMA studies that mention methamphetamine in eligibility criteria. This causes the discovery boost to occasionally fire on misattributed trial activity. The interactions-flag guard catches the worst cases, but more precise filtering (intervention-only trials) would help.

### 6.3 Future Work

In rough priority order:

- **Knowledge-graph embeddings** (DrugBank + CTD relations via TransE or ComplEx) — addresses cluster mismatch and direction blindness simultaneously by encoding relations explicitly rather than by text similarity.
- **Multi-class evidence labeling** (therapeutic / causal / marker / unstudied) — addresses the binary-classifier limitation that conflates "harmful" with "unrelated."
- **ClinicalTrials.gov outcome features** as model inputs — addresses ALS-style trial-outcome blindness; failed trials should not contribute to high model confidence.
- **Mechanism-aware features** — drug target protein annotations and disease pathway involvement, enabling true first-degree repurposing predictions.
- **Controlled-substance blacklist** — Schedule I / poisons / chemical weapons should be filtered regardless of preclinical literature; pairs the harm-aware reranker with a hard floor.
- **Cold-start train/test splits** in the official evaluation, reporting both pair-level AUC (for direct comparison to v1) and entity-disjoint AUC (the honest generalization number).

---

## 7. References

- DiMasi, J.A., Grabowski, H.G., & Hansen, R.W. (2016). Innovation in the pharmaceutical industry: New estimates of R&D costs. *Journal of Health Economics*, 47, 20-33.
- Zhang, Y., Chen, Q., Yang, Z., Lin, H., & Lu, Z. (2019). BioWordVec, improving biomedical word embeddings with subword information and MeSH. *Scientific Data*, 6(1), 52.
- Davis, A.P., et al. (2023). Comparative Toxicogenomics Database (CTD): update 2023. *Nucleic Acids Research*, 51(D1), D1257-D1262.
- Pushpakom, S., et al. (2019). Drug repurposing: progress, challenges and recommendations. *Nature Reviews Drug Discovery*, 18(1), 41-58.
- Haggett, D. (2024). BioLink: Knowledge Graph-Based Drug Repurposing with BioWordVec Embeddings. dustinhaggett.com/papers/biolink-paper.pdf
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K.Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*. (Temperature scaling for post-hoc calibration; Section 5.3.)
- Saerens, M., Latinne, P., & Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: A simple procedure. *Neural Computation*, 14(1), 21-41. (Bayesian prior correction via logit shift; Section 5.3.)
- Bordes, A., Usunier, N., Garcia-Duran, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS 2013*. (TransE; Section 6.3 future work.)
- Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex Embeddings for Simple Link Prediction. *ICML 2016*. (ComplEx; Section 6.3 future work.)
