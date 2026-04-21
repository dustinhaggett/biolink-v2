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
6. **Presentation** — Results are displayed with confidence badges, verdict badges, evidence quality indicators, pathway chains, and interactive follow-up Q&A

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
                    |   - Follow-up Q&A |
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
│   └── perplexity.py           # Evidence search, verdict/quality/pathway parsing, follow-up Q&A
├── explanation/
│   └── explainer.py            # Claude Haiku plain-English explanations
├── ui/
│   ├── components.py           # Result cards, badges, filters, search, compare, follow-up
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

## 5. References

- DiMasi, J.A., Grabowski, H.G., & Hansen, R.W. (2016). Innovation in the pharmaceutical industry: New estimates of R&D costs. *Journal of Health Economics*, 47, 20-33.
- Zhang, Y., Chen, Q., Yang, Z., Lin, H., & Lu, Z. (2019). BioWordVec, improving biomedical word embeddings with subword information and MeSH. *Scientific Data*, 6(1), 52.
- Davis, A.P., et al. (2023). Comparative Toxicogenomics Database (CTD): update 2023. *Nucleic Acids Research*, 51(D1), D1257-D1262.
- Pushpakom, S., et al. (2019). Drug repurposing: progress, challenges and recommendations. *Nature Reviews Drug Discovery*, 18(1), 41-58.
- Haggett, D. (2024). BioLink: Knowledge Graph-Based Drug Repurposing with BioWordVec Embeddings. dustinhaggett.com/papers/biolink-paper.pdf
