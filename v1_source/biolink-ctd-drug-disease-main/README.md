# AI-Driven Drug–Disease Modeling with CTD

A deep learning project comparing biomedical word embeddings (BioWordVec) and transformer sentence embeddings for predicting therapeutic drug–disease relationships using the Comparative Toxicogenomics Database (CTD).

## 📋 Overview

This project implements and compares two embedding-based approaches for predicting whether a chemical (drug) treats a given disease:

- **BioWordVec**: Domain-specific biomedical word embeddings pretrained on PubMed + MIMIC-III
- **Transformer Embeddings**: General-purpose sentence embeddings using `all-MiniLM-L6-v2`

Both models use the same CTD-derived dataset and MLP classifier architecture, enabling a fair comparison of domain-specific vs. general-purpose embeddings for biomedical NLP tasks.

## 🎯 Key Results

| Model | Accuracy | AUC | Average Precision | P@10 | P@50 | P@100 |
|-------|----------|-----|-------------------|------|------|-------|
| **BioWordVec** | 0.8761 | 0.9472 | 0.9488 | 1.0000 | 1.0000 | 1.0000 |
| **Transformer** | 0.8314 | 0.9128 | 0.9126 | 1.0000 | 1.0000 | 1.0000 |

### Key Insights

- **Domain-aware models** (BioWordVec) excel at global prediction metrics (AUC/AP)
- **Transformers** demonstrate excellent top-K ranking performance, useful for drug repurposing workflows
- Simple neural architectures + strong embeddings achieve high biomedical prediction accuracy

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch (with MPS support for Apple Silicon)
- Access to CTD data files (see Data section)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/biolink-ctd-drug-disease.git
cd biolink-ctd-drug-disease
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required data files:
   - CTD chemicals–diseases file: `CTD_chemicals_diseases.tsv.gz`
   - BioWordVec embeddings: `BioWordVec_PubMed_MIMICIII_d200.vec.bin`

   Place these in the `data/` directory or update paths in the notebook.

### Usage

Open and run the main notebook:
```bash
jupyter notebook notebooks/main.ipynb
```

Or use the Python modules:
```python
from src.data_loader import load_ctd_data, create_pairs
from src.embeddings import BioWordVecEmbedder, TransformerEmbedder
from src.models import train_model, evaluate_model

# Load and preprocess data
ctd_data = load_ctd_data("data/CTD_chemicals_diseases.tsv.gz")
pairs_df = create_pairs(ctd_data)

# Train BioWordVec model
bio_embedder = BioWordVecEmbedder("data/BioWordVec_PubMed_MIMICIII_d200.vec.bin")
# ... (see notebook for full workflow)
```

## 📁 Project Structure

```
biolink-ctd-drug-disease/
├── notebooks/
│   └── main.ipynb              # Main analysis notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # CTD data loading and preprocessing
│   ├── embeddings.py           # BioWordVec and Transformer embedders
│   ├── models.py               # MLP classifier and training utilities
│   └── evaluation.py           # Metrics and visualization functions
├── docs/
│   └── AI_Driven_Drug_Disease_Modeling_Using_Biomedical_and_Transformer_Embeddings.pdf
├── data/                       # Data directory (not tracked in git)
│   ├── .gitkeep
│   └── README.md               # Data download instructions
├── results/                    # Model outputs and visualizations
│   └── .gitkeep
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 📊 Dataset

The project uses the **Comparative Toxicogenomics Database (CTD)**:
- **Source**: [CTD Chemicals–Diseases](http://ctdbase.org/downloads/)
- **Filter**: Therapeutic relationships only (`DirectEvidence == "therapeutic"`)
- **Size**: ~39,265 positive pairs
- **Negative Sampling**: Random mismatching (1:1 ratio)

### Data Download

1. Download `CTD_chemicals_diseases.tsv.gz` from [CTD Downloads](http://ctdbase.org/downloads/)
2. Download BioWordVec embeddings from the [BioWordVec repository](https://github.com/ncbi-nlp/BioWordVec)
3. Place files in the `data/` directory

## 🔬 Methodology

1. **Data Preprocessing**
   - Filter CTD for therapeutic relationships
   - Generate negative samples via random mismatching
   - Stratified train/validation/test split (70/15/15)

2. **Embedding Generation**
   - **BioWordVec**: Tokenize → retrieve embeddings → average pooling
   - **Transformer**: Direct sentence encoding with `sentence-transformers`

3. **Feature Engineering**
   - Interaction features: `[chem_emb, disease_emb, |chem - disease|, chem * disease]`

4. **Model Architecture**
   - MLP: Linear(4d → 256) → BatchNorm → ReLU → Dropout(0.3) → Linear(256 → 1)
   - Loss: BCEWithLogitsLoss
   - Optimizer: Adam (lr=1e-3)

5. **Evaluation Metrics**
   - Accuracy, AUC-ROC, Average Precision
   - Precision@K (K ∈ {10, 50, 100, 200})
   - ROC and Precision-Recall curves

## 📈 Results & Visualizations

The notebook includes:
- ROC curves comparing both models
- Precision-Recall curves
- EDA of most common chemicals and diseases
- Top predicted drug repurposing candidates

## 🔍 Drug Repurposing Discovery

The project includes a proof-of-concept repurposing screen:
- Scores unseen chemical–disease combinations
- Ranks candidates by predicted therapeutic probability
- Surfaces high-confidence novel therapeutic relationships

Example predictions:
- Etoposide → Melanoma
- Physostigmine → Bradycardia
- Baclofen → Tremor

## 🛠️ Technologies

- **PyTorch**: Deep learning framework
- **scikit-learn**: Data splitting and metrics
- **Gensim**: BioWordVec embeddings
- **sentence-transformers**: Transformer embeddings
- **pandas/numpy**: Data manipulation
- **matplotlib**: Visualizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CTD**: Comparative Toxicogenomics Database
- **BioWordVec**: Biomedical word embeddings by NCBI
- **sentence-transformers**: Hugging Face sentence transformer models

## 📄 Documentation

A detailed paper/report is available in the [`docs/`](docs/) directory:
- **Paper**: `AI_Driven_Drug_Disease_Modeling_Using_Biomedical_and_Transformer_Embeddings.pdf`

This document provides comprehensive methodology, results analysis, and discussion of the findings.

## 📧 Contact

For questions or suggestions, please open an issue or contact the repository maintainer.

---

**Note**: This project is part of a machine learning portfolio demonstrating biomedical NLP and drug discovery applications.

