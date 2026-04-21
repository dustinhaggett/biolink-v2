# BioLink

Drug repurposing discovery tool powered by knowledge graph AI. Search by disease to find ranked drug candidates, or search by drug to find conditions it may treat.

Built for AAI 595 (Applied ML) at Drexel University.

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

These files are too large for git and must be downloaded separately:

| File | Size | Location | Source |
|------|------|----------|--------|
| BioWordVec embeddings | ~13 GB | `data/BioWordVec_PubMed_MIMICIII_d200.vec.bin` | [NCBI/NLM](https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin) |
| Model weights | ~810 KB | `models/biolink_v1.pt` | Contact repo owner |
| CTD data (optional) | ~152 MB | `data/CTD_chemicals_diseases.tsv.gz` | [CTD downloads](https://ctdbase.org/downloads/) |

Download the BioWordVec file:
```bash
curl -L -o data/BioWordVec_PubMed_MIMICIII_d200.vec.bin \
  https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.vec.bin
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

## Disclaimer

BioLink is a hypothesis-generation tool for research purposes only. It is **not medical advice**. Always consult a qualified healthcare provider before considering any treatment.
