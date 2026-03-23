# Data Directory

This directory contains the data files required for the project.

## Required Files

### 1. CTD Chemicals–Diseases File
- **File**: `CTD_chemicals_diseases.tsv.gz`
- **Source**: [CTD Downloads](http://ctdbase.org/downloads/)
- **Description**: Comparative Toxicogenomics Database chemical–disease associations
- **Size**: ~971M rows (uncompressed)

### 2. BioWordVec Embeddings
- **File**: `BioWordVec_PubMed_MIMICIII_d200.vec.bin`
- **Source**: [BioWordVec GitHub](https://github.com/ncbi-nlp/BioWordVec)
- **Description**: Biomedical word embeddings pretrained on PubMed and MIMIC-III
- **Dimensions**: 200
- **Vocabulary**: ~16.5M words

## Download Instructions

1. **CTD Data**:
   ```bash
   # Download from CTD website
   wget http://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz
   ```

2. **BioWordVec**:
   ```bash
   # Download from BioWordVec repository
   wget https://github.com/ncbi-nlp/BioWordVec/releases/download/v1.0/BioWordVec_PubMed_MIMICIII_d200.vec.bin
   ```

## Note

These files are large and are not tracked in git. Please download them separately and place them in this directory.

