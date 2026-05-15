---
title: BioLink v2
emoji: 🧬
colorFrom: green
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
---

# BioLink

Drug repurposing discovery tool powered by knowledge graph AI. Search by disease to find ranked drug candidates, or search by drug to find conditions it may treat.

Built for AAI 595 (Applied ML) at Stevens Institute of Technology.

## Demo

- **Live app:** https://huggingface.co/spaces/dustinhaggett/biolink-v2
- **Video walkthrough:** [docs/BioLink_Demo.mp4](docs/BioLink_Demo.mp4)
- **Project report:** [docs/PROJECT_REPORT.pdf](docs/PROJECT_REPORT.pdf)

## Features

- **Disease search** — Enter a disease, get ranked drug repurposing candidates with calibrated confidence scores
- **Drug search** — Enter a drug to discover what conditions it may treat (reverse search)
- **Live evidence** — Perplexity-powered evidence search with verdict badges, evidence quality ratings, and pathway chains
- **Clinical trials** — Automatic lookup of relevant trials on ClinicalTrials.gov
- **Compare mode** — Select multiple candidates for side-by-side comparison
- **Ask follow-up questions** — Per-result Q&A grounded in published evidence
- **Export** — Download results as CSV or PDF report
- **Batch mode** — Upload a CSV of diseases for bulk predictions

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/dustinhaggett/biolink-v2.git
cd biolink-v2
```

### 2. Install dependencies

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

### 3. Download large data files

The trained model weights (`models/biolink_v1.pt`) and cached entity embeddings (`data/drug_embeddings.npy`, `data/disease_embeddings.npy`) are already in the repo, so the app can run out of the box. Two large source files are excluded from git and only needed if you want to retrain the model or rebuild the embedding caches from scratch:

| File | Size | Location | Source |
|------|------|----------|--------|
| BioWordVec embeddings | ~13 GB | `data/BioWordVec_PubMed_MIMICIII_d200.vec.bin` | [NCBI/NLM](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin) |
| CTD chemicals-diseases | ~152 MB | `data/CTD_chemicals_diseases.tsv.gz` | [CTD downloads](https://ctdbase.org/downloads/) |

Download BioWordVec:
```bash
curl -L -o data/BioWordVec_PubMed_MIMICIII_d200.vec.bin \
  https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin
```

Download CTD:
```bash
curl -L -o data/CTD_chemicals_diseases.tsv.gz \
  https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz
```

### 4. API keys (optional)

The app works without API keys but with reduced functionality:

- **Without keys**: Disease/drug matching uses fuzzy string matching instead of Claude. Evidence search and follow-up questions are disabled.
- **With keys**: Full LLM-powered intent mapping, live evidence search, and follow-up Q&A.

To add keys, create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
PERPLEXITY_API_KEY = "pplx-..."
```

### 5. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Reproducing Training From Scratch

The committed model weights and cached embeddings let the app run without retraining. If you want to reproduce training end-to-end:

```bash
# 1. Train the MLP and produce val_logits / val_labels / model weights
python scripts/generate_v1_artifacts.py \
    --ctd data/CTD_chemicals_diseases.tsv.gz \
    --biowordvec data/BioWordVec_PubMed_MIMICIII_d200.vec.bin

# 2. Cache per-entity BioWordVec embeddings for fast inference
python scripts/cache_embeddings.py

# 3. Fit temperature scaling and produce the reliability diagram
python scripts/fit_temperature.py
```

The underlying training code lives in [`v1_source/biolink-ctd-drug-disease-main/`](v1_source/biolink-ctd-drug-disease-main/) — see its `README.md` for additional context and the original methodology. On an Apple Silicon laptop the full pipeline runs end-to-end in roughly 15-25 minutes; the bottleneck is loading the 13 GB BioWordVec embeddings.

## Project Structure

```
biolink-v2/
├── app.py                  # Streamlit app entry point
├── core/
│   ├── model.py            # BioLinkModel (MLP + BioWordVec embeddings)
│   ├── inference.py        # disease_to_drugs / drug_to_diseases pipelines
│   ├── intent_mapper.py    # Disease/drug entity resolution (Claude + fuzzy)
│   └── calibration.py      # Temperature scaling for confidence scores
├── enrichment/
│   ├── runner.py           # Async enrichment orchestrator
│   ├── pubmed.py           # PubMed co-occurrence counts
│   ├── openfda.py          # FDA approval status lookup
│   ├── clinicaltrials.py   # ClinicalTrials.gov trial finder
│   └── perplexity.py       # Perplexity evidence search + follow-up
├── explanation/
│   └── explainer.py        # Claude-powered plain-English explanations
├── ui/
│   ├── components.py       # Streamlit UI components
│   ├── styles.py           # CSS injection
│   └── pdf_export.py       # PDF report generator
├── data/
│   ├── drugs_list.txt      # 7,163 CTD drug names
│   ├── diseases_list.txt   # 2,525 CTD disease names
│   └── temperature.json    # Calibration parameter
├── models/
│   └── biolink_v1.pt       # Trained MLP weights (not in git)
└── requirements.txt
```

## AI Tools Usage

This project leverages AI tools extensively, as encouraged by the course guidelines:

- **Claude Code (Anthropic)** — Primary development tool. Used for architecture design, code generation, debugging, feature implementation, and documentation. Claude Code accelerated development of all major features including the Perplexity integration, clinical trials finder, reverse search, PDF export, and UI components.
- **Claude API (Haiku)** — Runtime component. Powers disease/drug entity resolution (intent mapping) and plain-English explanation generation within the app itself.
- **Perplexity Sonar API** — Runtime component. Provides live evidence search grounding each drug-disease prediction in published literature, clinical trials, and mechanism-of-action data.
- **GitHub Copilot** — Used for code completion during early development phases.

## Disclaimer

BioLink is a hypothesis-generation tool for research purposes only. It is **not medical advice**. Always consult a qualified healthcare provider before considering any treatment.
