# BioLink v2 — Technical Specification

**Version:** 1.0
**Status:** Ready for implementation
**Stack:** Python · PyTorch · scikit-learn · Claude API · Streamlit · Hugging Face Spaces
**Local dev target:** Apple M3 Max, 128GB unified memory — use `torch.device("mps")` for local runs

---

## 0. Overview

BioLink v2 wraps the existing v1 MLP model (AUC=0.947) in a consumer-facing drug repurposing pipeline. A user types a disease in plain English; the system returns a ranked list of candidate drugs with calibrated confidence tiers, FDA approval status, PubMed evidence counts, and LLM-generated plain-English explanations.

**No model retraining.** All novel work is: calibration, disease-first inference, LLM intent mapping, enrichment APIs, explanation layer, and UI redesign.

---

## 1. Repository Layout

```
biolink_v2/
├── app.py                  # Streamlit entrypoint
├── requirements.txt
├── .env.example            # ANTHROPIC_API_KEY placeholder
│
├── core/
│   ├── __init__.py
│   ├── model.py            # Load v1 model weights, BioWordVec encoder, MLP forward pass
│   ├── calibration.py      # Temperature scaling: fit T on val set, calibrated_proba()
│   ├── inference.py        # disease_to_drugs(): main pipeline entry point
│   └── intent_mapper.py    # Free-text -> CTD disease entity via Claude API
│
├── enrichment/
│   ├── __init__.py
│   ├── pubmed.py           # async PubMed E-utilities: evidence_count(drug, disease)
│   ├── openfda.py          # async OpenFDA: fda_status(drug_name)
│   └── runner.py           # asyncio.gather() over top-N results
│
├── explanation/
│   ├── __init__.py
│   └── explainer.py        # Claude API: generate plain-English mechanism explanation
│
├── ui/
│   ├── components.py       # Reusable Streamlit widgets (result card, confidence badge, disclaimer)
│   └── styles.py           # CSS injection for visual polish
│
├── data/
│   ├── drugs_list.txt      # All CTD drug names (one per line, from v1 dataset)
│   ├── diseases_list.txt   # All CTD disease names (one per line, from v1 dataset)
│   └── temperature.json    # {"T": <float>} — persisted after calibration fitting
│
├── models/
│   └── biolink_v1.pt       # v1 MLP weights (copied from v1 repo)
│
├── scripts/
│   ├── fit_temperature.py  # One-time script: fit T on val set, write data/temperature.json
│   └── validate_calibration.py  # ECE before/after, reliability diagram plot
│
└── tests/
    ├── test_model.py
    ├── test_calibration.py
    ├── test_intent_mapper.py
    ├── test_enrichment.py
    └── test_explainer.py
```

---

## 2. Data & Assets

### 2.1 Inputs Required (from v1)
- `models/biolink_v1.pt` — saved PyTorch MLP state dict
- `data/drugs_list.txt` — all drug names from the CTD training set (needed for scoring all drugs against a query disease)
- `data/diseases_list.txt` — CTD disease names (used by intent mapper for entity alignment)
- BioWordVec model binary — either bundled or downloaded at runtime from the v1 source

### 2.2 Validation Set
- Required for temperature scaling: a held-out split from v1 (logits + true labels)
- If not already saved, re-run v1 inference on the val set and save `data/val_logits.npy` + `data/val_labels.npy`

---

## 3. Core Layer

### 3.1 `core/model.py`

```python
class BioLinkModel:
    def __init__(self, weights_path, biowordvec_path):
        # Load MLP from v1 architecture (mirror v1 exactly)
        # Load BioWordVec KeyedVectors
        pass

    def encode_disease(self, ctd_disease_name: str) -> np.ndarray:
        # BioWordVec encode -> 200-dim vector
        pass

    def encode_drug(self, drug_name: str) -> np.ndarray:
        # BioWordVec encode -> 200-dim vector
        pass

    def feature_vector(self, drug_vec, disease_vec) -> np.ndarray:
        # [drug, disease, |drug - disease|, drug * disease] -> 800-dim
        # Note: diff is np.abs(drug_vec - disease_vec), matching v1 build_pair_features exactly
        pass

    def raw_logit(self, drug_name: str, disease_vec: np.ndarray) -> float:
        # Returns pre-sigmoid logit for a single drug
        pass

    def score_all_drugs(self, disease_vec: np.ndarray) -> list[tuple[str, float]]:
        # Returns [(drug_name, raw_logit), ...] for all drugs in drugs_list
        # Sorted descending by logit
        pass
```

**v1 MLP architecture (exact — do not deviate):**
```python
# MLPClassifier(input_dim=800, hidden_dim=256, dropout=0.3)
nn.Sequential(
    nn.Linear(800, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1),
)
# forward: return self.net(x).squeeze(-1)  # raw logit, no sigmoid
```
- Input dim is always 800 (4 × 200-dim BioWordVec)
- `score_all_drugs` returns raw logits — calibration handles sigmoid
- BioWordVec binary filename: `BioWordVec_PubMed_MIMICIII_d200.vec.bin`
- Tokenizer (from v1 `embeddings.py`): `re.compile(r"[A-Za-z0-9\-]+")`, lowercased, mean-pooled; OOV → zero vector

### 3.2 `core/calibration.py`

Temperature scaling post-hoc calibration.

```python
class TemperatureScaler:
    def __init__(self, T: float = 1.0):
        self.T = T

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        # Minimize NLL over val set w.r.t. T using scipy.optimize.minimize_scalar
        # Save T to data/temperature.json
        # Return fitted T
        pass

    def calibrated_proba(self, logit: float) -> float:
        # sigmoid(logit / T)
        pass

    @classmethod
    def load(cls, path="data/temperature.json") -> "TemperatureScaler":
        # Load T from JSON
        pass

def confidence_tier(proba: float) -> str:
    # Strong:      proba >= 0.80
    # Moderate:    proba >= 0.50
    # Speculative: proba < 0.50
    pass
```

**Fitting script** (`scripts/fit_temperature.py`):
1. Load `data/val_logits.npy` and `data/val_labels.npy`
2. Instantiate `TemperatureScaler`, call `.fit()`
3. Print T, print ECE before/after
4. Write `data/temperature.json`

This script runs once. Commit `temperature.json` to the repo.

### 3.3 `core/intent_mapper.py`

Maps free-text disease input to the best matching CTD disease entity.

```python
def map_disease(user_input: str, candidate_diseases: list[str]) -> dict:
    """
    Returns:
        {
            "ctd_entity": str,       # matched CTD disease name
            "confidence": str,       # "high" | "medium" | "low"
            "display_name": str,     # clean name for UI
            "clarification": str | None  # if low confidence, a question to ask user
        }
    """
    pass
```

**Prompt design:**
- System: "You are a biomedical entity resolver. Given a user's plain-English disease description, identify the single best matching disease entity from the provided list of CTD disease names. Return JSON only."
- User: `f"User input: '{user_input}'\n\nCTD disease list (select one):\n{formatted_candidates}"`
- Response schema: `{"ctd_entity": str, "confidence": "high"|"medium"|"low", "display_name": str, "clarification": str|null}`
- Model: `claude-opus-4-6`
- Temperature: 0 (deterministic)
- Parse with `json.loads()`, validate keys

**Edge cases:**
- If confidence is "low", surface `clarification` to user before proceeding
- If no good match, return `{"ctd_entity": null, "clarification": "Could you describe your condition differently?"}`

### 3.4 `core/inference.py`

Main pipeline orchestrator.

```python
def disease_to_drugs(
    user_input: str,
    top_n: int = 20,
    model: BioLinkModel = None,
    scaler: TemperatureScaler = None,
) -> dict:
    """
    Returns:
        {
            "query": str,                    # user's original input
            "ctd_entity": str,               # resolved CTD disease name
            "display_name": str,
            "results": [
                {
                    "drug": str,
                    "logit": float,
                    "proba": float,
                    "tier": str,             # Strong / Moderate / Speculative
                    # enrichment fields added later:
                    "pubmed_count": int | None,
                    "fda_status": str | None,
                    "explanation": str | None,
                },
                ...
            ]
        }
    """
    pass
```

**Pipeline steps:**
1. Call `intent_mapper.map_disease(user_input, diseases_list)`
2. If null entity or low confidence with clarification → return early with clarification prompt
3. `disease_vec = model.encode_disease(ctd_entity)`
4. `scored = model.score_all_drugs(disease_vec)` → top_n results
5. Apply `scaler.calibrated_proba()` to each logit
6. Assign confidence tiers
7. Return structured dict (enrichment is added async in next step)

---

## 4. Enrichment Layer

### 4.1 `enrichment/pubmed.py`

```python
async def evidence_count(drug: str, disease: str, session: aiohttp.ClientSession) -> int:
    """
    Query PubMed E-utilities esearch endpoint.
    Search term: f"{drug}[Title/Abstract] AND {disease}[Title/Abstract]"
    Returns: count (int), or 0 on failure
    """
    pass
```

**API:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
**Params:** `db=pubmed&term=...&retmode=json&retmax=0`
**Parse:** `result["esearchresult"]["count"]`
**Timeout:** 5s per request. Return 0 on timeout/error (non-blocking).

### 4.2 `enrichment/openfda.py`

```python
async def fda_status(drug_name: str, session: aiohttp.ClientSession) -> str:
    """
    Query OpenFDA drug/label endpoint.
    Returns: "FDA Approved" | "Not in FDA Database" | "Unknown"
    """
    pass
```

**API:** `https://api.fda.gov/drug/label.json`
**Params:** `search=openfda.brand_name:"{drug}" OR openfda.generic_name:"{drug}"&limit=1`
**Parse:** If results exist → "FDA Approved"; 404 → "Not in FDA Database"; error → "Unknown"
**Timeout:** 5s per request.

### 4.3 `enrichment/runner.py`

```python
async def enrich_results(results: list[dict]) -> list[dict]:
    """
    Runs PubMed + OpenFDA lookups in parallel for all results.
    Mutates and returns results with pubmed_count and fda_status filled in.
    """
    async with aiohttp.ClientSession() as session:
        tasks = []
        for r in results:
            tasks.append(evidence_count(r["drug"], ctd_entity, session))
            tasks.append(fda_status(r["drug"], session))
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        # zip back into results
    pass

def run_enrichment(results: list[dict], ctd_entity: str) -> list[dict]:
    # Sync wrapper: asyncio.run(enrich_results(...))
    pass
```

**Rate limiting:** Add 100ms delay between PubMed batches if >20 results to avoid NCBI throttling.

---

## 5. Explanation Layer

### 5.1 `explanation/explainer.py`

```python
def explain_prediction(
    drug: str,
    disease: str,
    proba: float,
    tier: str,
    pubmed_count: int,
    fda_status: str,
) -> str:
    """
    Returns a 2-4 sentence plain-English explanation.
    """
    pass
```

**Prompt design:**

System prompt:
```
You are a biomedical science communicator helping patients and clinicians understand
drug repurposing predictions. Write clear, honest, non-alarmist explanations.
Always include an appropriate medical disclaimer. Never recommend treatment.
```

User prompt:
```
Drug: {drug}
Disease: {disease}
Model confidence: {tier} ({proba:.0%})
PubMed evidence: {pubmed_count} publications
FDA status: {fda_status}

Write a 2-4 sentence explanation of:
1. The known or plausible mechanism linking this drug to this disease
2. What the evidence level means
3. A brief disclaimer

Keep it accessible to a non-specialist. Be honest about uncertainty.
```

**Model:** `claude-opus-4-6`
**Max tokens:** 200
**Temperature:** 0.3

**Batching:** Generate explanations only for top 10 results (not all 20) to manage latency and API cost.

---

## 6. UI Layer

### 6.1 App Flow (`app.py`)

```
[Disease input box]
    -> [Submit]
        -> intent mapping (with spinner)
        -> if clarification needed: show clarification prompt, re-prompt user
        -> scoring (with spinner)
        -> enrichment parallel async (with spinner)
        -> explanation for top 10 (with spinner)
        -> render results
```

### 6.2 Result Card Layout

Each result card displays:
- Drug name (large, bold)
- Confidence badge: colored pill — Strong (green) / Moderate (yellow) / Speculative (gray)
- Calibrated probability (e.g., "82% confidence")
- FDA status badge
- PubMed count (e.g., "14 publications")
- Expandable section: LLM explanation (collapsed by default, click to expand)

### 6.3 Disclaimer

Persistent banner at top and bottom:
> **Not medical advice.** BioLink v2 is a hypothesis-generation tool for research purposes only. Always consult a qualified physician before considering any treatment.

### 6.4 `ui/components.py`

```python
def render_disclaimer(): ...
def render_result_card(result: dict): ...
def render_confidence_badge(tier: str, proba: float): ...
def render_search_box() -> str: ...  # returns user input
def render_clarification(question: str) -> str: ...
def render_spinner_context(label: str): ...  # context manager wrapping st.spinner
```

### 6.5 States

The app has four states managed via `st.session_state`:
1. `idle` — initial load, show search box
2. `clarifying` — low-confidence entity match, show clarification question
3. `loading` — pipeline running, show spinners
4. `results` — display result cards

---

## 7. Environment & Configuration

**`.env.example`:**
```
ANTHROPIC_API_KEY=your_key_here
```

**`requirements.txt` (pinned):**
```
torch>=2.0
gensim>=4.3          # BioWordVec loading
scikit-learn>=1.3
scipy>=1.11
anthropic>=0.30
aiohttp>=3.9
streamlit>=1.35
numpy>=1.24
python-dotenv>=1.0
```

**Loading order in `app.py`:**
1. `load_dotenv()`
2. Load BioWordVec (cached with `@st.cache_resource`)
3. Load MLP weights (cached with `@st.cache_resource`)
4. Load temperature scaler (from `data/temperature.json`)
5. Load drugs/diseases lists

Use `@st.cache_resource` for all heavy assets so they load once per session.

---

## 8. Error Handling

| Scenario | Behavior |
|---|---|
| Intent mapper returns null entity | Show: "I couldn't identify a disease matching your description. Try rephrasing — e.g., 'atrial fibrillation' or 'rapid heartbeat'." |
| Claude API timeout / error | Fall back to fuzzy string match against diseases_list using `difflib.get_close_matches()` |
| PubMed / OpenFDA timeout | Show `pubmed_count=None` → display "Evidence data unavailable" in card |
| Explanation API failure | Show card without explanation; log error silently |
| BioWordVec OOV (out-of-vocab) | For OOV tokens, use zero vector; log warning |

---

## 9. Testing

### Unit tests (pytest)

| File | Tests |
|---|---|
| `test_model.py` | encode_disease returns 200-dim; feature_vector is 800-dim; score_all_drugs returns sorted list |
| `test_calibration.py` | calibrated_proba is in [0,1]; T > 0; confidence_tier boundaries correct |
| `test_intent_mapper.py` | Mock Claude API; test high/medium/low confidence paths; test null entity path |
| `test_enrichment.py` | Mock aiohttp; test PubMed count parsing; test OpenFDA status parsing; test timeout fallback |
| `test_explainer.py` | Mock Claude API; test output is non-empty string; test disclaimer present |

Run with: `pytest tests/ -v`

---

## 10. Deployment (Hugging Face Spaces)

- Space type: Streamlit
- Hardware: CPU Basic (free tier sufficient)
- Secrets: `ANTHROPIC_API_KEY` set in Space secrets (not in repo)
- BioWordVec: Either commit a compressed version or download at startup from a stable URL with `@st.cache_resource`
- `README.md` header with `title`, `emoji`, `colorFrom`, `colorTo`, `sdk: streamlit`, `app_file: app.py` for Spaces metadata

**`.gitignore`:**
```
.env
models/*.pt      # if weights are large, use Git LFS
data/val_logits.npy
data/val_labels.npy
__pycache__/
*.pyc
```

---

## 11. Implementation Order (Codex Build Sequence)

Build in this order to enable testing at each step:

0. **Prerequisite — generate val set artifacts:** Re-run v1 notebook with CTD data. Save `data/val_logits.npy`, `data/val_labels.npy`, `data/drugs_list.txt`, `data/diseases_list.txt`, and `models/biolink_v1.pt`. Also download `BioWordVec_PubMed_MIMICIII_d200.vec.bin`. Nothing else can proceed without these.

1. **`core/model.py`** — port v1 architecture, verify weights load, verify `score_all_drugs` output shape
2. **`scripts/fit_temperature.py` + `core/calibration.py`** — fit T, validate ECE improvement, commit `temperature.json`
3. **`core/intent_mapper.py`** — Claude API intent mapping, test with 5-10 manual examples
4. **`core/inference.py`** — wire model + intent mapper + calibration into end-to-end pipeline
5. **`enrichment/pubmed.py` + `enrichment/openfda.py` + `enrichment/runner.py`** — async enrichment, test with known drug names
6. **`explanation/explainer.py`** — prompt design, test explanation quality on 5 examples
7. **`ui/components.py` + `app.py`** — full Streamlit UI with all four states
8. **`tests/`** — unit tests for all modules
9. **Deploy** — Hugging Face Spaces

---

## 12. Open Questions / Decisions for Review

- [x] **BioWordVec binary:** `BioWordVec_PubMed_MIMICIII_d200.vec.bin` — confirmed downloaded to `data/` (~13GB actual size, not 2.3GB as documented online). **HF Spaces note:** This file is too large for standard HF Spaces — must be hosted as a HF Hub model repo artifact and streamed via `huggingface_hub.hf_hub_download()` at startup, or quantized/cached. Address at deploy time.
- [x] **Val set availability:** NOT saved in v1 — all runs in-memory in the notebook. **Step 0 (before anything else):** re-run v1 notebook on the CTD data to regenerate the val split, save `data/val_logits.npy` and `data/val_labels.npy`. Use same seed=42 and split params (test_size=0.30, val_size=0.50) from `data_loader.create_splits()`.
- [x] **Drugs list size:** 7,164 unique drugs. Numpy-batched scoring will be instant on M3 Max. Pre-embed all 7,164 drug vectors at startup and cache as a (7164, 200) matrix.
- [x] **Top-N for explanations:** Set to top 10. Adjust if API cost is a concern.
- [x] **Clarification UX:** Block + ask on low-confidence intent match. Wrong disease entity = completely wrong rankings. Extra click is worth it.
