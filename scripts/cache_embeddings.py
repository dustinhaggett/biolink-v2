"""Pre-compute and cache all drug/disease embeddings so deployment
doesn't need the 13 GB BioWordVec binary."""

from pathlib import Path
import numpy as np
from core.model import BioLinkModel

DATA = Path("data")

model = BioLinkModel(
    weights_path="models/biolink_v1.pt",
    biowordvec_path=str(DATA / "BioWordVec_PubMed_MIMICIII_d200.vec.bin"),
    drugs_list_path=str(DATA / "drugs_list.txt"),
    diseases_list_path=str(DATA / "diseases_list.txt"),
)

np.save(DATA / "drug_embeddings.npy", model.drug_embeddings)
print(f"Saved drug embeddings: {model.drug_embeddings.shape}")

np.save(DATA / "disease_embeddings.npy", model.disease_embeddings)
print(f"Saved disease embeddings: {model.disease_embeddings.shape}")
