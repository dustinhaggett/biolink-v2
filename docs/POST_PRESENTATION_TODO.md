# Post-Presentation TODO — Model Fixes

Captured during pre-presentation review (2026-04-30), revised post-presentation (2026-05-01). Final project due in ~2 weeks.

## Progress log

**2026-05-01 morning session:**
- ✅ **Day 1 — Regression test suite locked.** [scripts/run_regression.py](../scripts/run_regression.py) runs 12 documented queries (6 wins + 6 failures) against the model directly (bypasses LLM intent mapper for determinism). Baseline saved to `results/regression_baseline_20260501_093259.json`. Sub-second per run.
- ✅ **Day 11 — Recalibration done early** (moved up because score saturation was even worse than expected — every result 0.97–1.00). Implemented `prior_shift` in [core/calibration.py](../core/calibration.py); set p_real=1% (shift ≈ 4.595). Tier thresholds adjusted: Strong ≥ 0.30, Moderate ≥ 0.10. Tests updated. Result: 240/240 → 197/240 Strong (43 Moderate now visible). Ranking preserved exactly.
- 🔍 **Discovery during recalibration:** top-20 cutoff is hiding lots of signal. See "Candidate-count finding" section below.
- ✅ **Days 2-3 — Honest baselines computed.** [scripts/eval_baselines.py](../scripts/eval_baselines.py) evaluates per-disease ranking AUC vs trivial baselines on all 2,526 diseases. **The killer paper finding** (see "Honest baselines" section below).

**2026-05-01 second session (Days 4-7):**
- ⚠️ **Hard-negative retraining done but reveals a deeper finding** (see "Hard-negative trade-off" section below). Implemented [scripts/retrain_with_hard_negatives.py](../scripts/retrain_with_hard_negatives.py). Trained 3 variants (100% hard, 50/50 mix, 25/75 mix). Each variant fixes some failures but degrades some wins. **No mix is unambiguously better.**
- ⚠️ **Decision: do NOT ship retrained model as production yet.** The Perplexity-judge work (Days 8-9) is the better mechanism for safety re-ranking. Hard negatives can complement but don't replace it.
- ✅ **Days 8-9 — Harm-aware reranking done.** Added `HARM_FOR_INDICATION: HARMFUL/NOT_HARMFUL/UNKNOWN` to Perplexity's structured output ([enrichment/perplexity.py](../enrichment/perplexity.py)). Built [core/reranking.py](../core/reranking.py) — pure function, harm-only, never demotes on absence of evidence. Wired into [app.py](../app.py) `_run_pipeline`. **21 new reranking tests, all pass.** Locks in the discovery-vs-harm invariants (e.g., `test_insufficient_evidence_unknown_harm_is_unchanged`).

## Guiding principle: discovery vs. harm

**Drug repurposing is a discovery tool.** The most valuable predictions are unexpected uses that don't have strong evidence yet — that's the whole point. A naïve safety filter ("only surface validated treatments") would kill the entire purpose of the tool.

The fix must distinguish:
- **Genuinely unstudied pairs** → preserve. These are the discovery space.
- **Studied and known to be harmful for THIS specific indication** → demote. These are bias, not insight.

Cocaine for alcoholism is not a "bold novel hypothesis we should respect" — it's the model failing to distinguish "co-occurs with alcoholism in addiction literature" from "treats alcoholism." But cocaine for cancer (with no studied therapeutic data either way) IS the discovery space — we don't want to demote it just because cocaine is generally harmful.

## Failure modes we documented

| Disease | What we saw | Failure mode |
|---|---|---|
| Lyme Disease | Cyclosporine, methylprednisolone, prednisone in top 10 | Cluster mismatch — Lyme's embedding sits near autoimmune diseases |
| Type 2 Diabetes | **Streptozocin** at #10 (induces diabetes); **Nicotine** at #19 | Direction blindness |
| Alcoholism | **Top 5: Cocaine, Heroin, PCP, Methamphetamine** at 100% Strong | Direction blindness + popularity bias |
| Insomnia | **Nicotine at #1** (a stimulant) | Direction blindness |
| Amnesia | **Scopolamine #1** (induces amnesia), Strychnine, PCP in top 20 | Direction blindness — model can't distinguish induce vs. treat |
| ADHD | Methylphenidate (the primary ADHD drug) buried at #8 under CBD, Nicotine, Scopolamine | Cluster mismatch + direction blindness |
| ALS | 19/20 are failed-trial drugs; only Riluzole (#6) is correct | "Trialed ≠ effective" |
| Huntington's | Tetrabenazine (only HD-specific drug) at #20; rest are Parkinson's drugs | Cluster mismatch |

## Honest baselines finding (2026-05-01) — paper killer

Reproduced v1's reported AUC of 0.947 on the 50/50 pair-level held-out set (got 0.9497, matches). Then computed **per-disease ranking AUC** — the realistic task: rank all 7,163 drugs for each disease, mark CTD therapeutics as positives. Same data, more honest framing.

### Per-disease ranking AUC across all 2,526 diseases:

| Method | AUC mean | AUC median | What it is |
|---|---|---|---|
| **Trained MLP** | **0.9470** | **0.9770** | The deployed model |
| **Drug popularity** | **0.8803** | **0.9083** | Rank drugs by global # therapeutic indications. Same ranking for every disease. |
| Cosine similarity | 0.8008 | 0.8420 | Plain BioWordVec cosine sim. No ML at all. |
| Random | 0.5099 | 0.5072 | Sanity check ✓ |

**The MLP only adds +6.7 points (mean) above a popularity baseline that uses the same drug ranking for every disease.** ~93% of the model's apparent performance is captured by global drug popularity.

### The story gets worse:

- **MLP only beats popularity by >0.05 on 45.7% of diseases** (1,155 of 2,526). For the other 54%, popularity is essentially as good or better.
- **Popularity actively beats MLP on 16.9% of diseases** (428 of 2,526) — typically rare conditions with only 1-2 known therapeutics.

### And worse: AUC masks user-facing failures

- **Lyme Disease MLP AUC = 0.999** (only 2 positives among 7,164 candidates makes ranking trivial).
- **But the #1 ranked drug for Lyme is Cyclosporine** — wrong and harmful (immunosuppressant for a bacterial infection).
- AUC near 1.0 doesn't mean the user sees correct top results. **Precision@1 / @5 is what users actually experience.**

### Paper framing

> *"v1 reported an AUC of 0.947 on a 50/50 pair-level test set. We reproduced that number (0.9497). However, when we evaluate the practical task — per-disease ranking of all 7,163 candidate drugs — we find that a trivial baseline that ranks drugs by their global frequency in CTD therapeutic associations achieves 0.880 AUC. The MLP contributes only +6.7 points of disease-specific signal beyond this popularity prior. Furthermore, AUC near 1.0 can coexist with a harmful #1 recommendation: for example, our model achieves AUC=0.999 on Lyme Disease but ranks cyclosporine — a contraindicated immunosuppressant — as the top therapeutic candidate."*

### Files

- [scripts/eval_baselines.py](../scripts/eval_baselines.py) — re-runnable eval. `--regression-only` for fast testing, `--all-diseases` for full report.
- `results/baselines_20260501_095213.json` — full per-disease results across 2,526 diseases.

### Implications for the plan

- The popularity baseline at 0.88 confirms **popularity bias is the dominant failure mode** in the model. Hard-negative training (P0) needs to specifically address this — not just direction blindness.
- **Add precision@k to the regression suite output.** AUC is too forgiving for the user-facing ranking task.
- The paper now has a quantitative story to tell, not just qualitative failure cases.

## Hard-negative trade-off finding (2026-05-01) — second paper-worthy result

We retrained the v1 MLP with marker/mechanism pairs as hard negatives — exactly as the original plan called for — but the result was more nuanced than expected. **Hard negatives don't simply "fix" the failures; they introduce a different bias.**

### What we tried

Same v1 architecture, hyperparameters, seed. Only difference: replace some/all random negatives with CTD `marker/mechanism` pairs. Three variants:

| Variant | Negative composition | Val AUC | Wins preserved | Failures improved |
|---|---|---|---|---|
| Baseline (v1) | 39,516 random | 0.9497 | 6/6 | 0/6 |
| **100% hard** | 39,516 marker/mechanism | 0.9161 | **0/6** | 5/6 |
| **50/50 mix** | 19,758 hard + 19,758 random | 0.9005 | 3/6 | 5/6 |
| **25/75 mix** | 9,879 hard + 29,637 random | 0.9?? | 4/6 | 5/6 |

### What 100% hard negatives did

**Inverse popularity bias.** Every disease's top 10 became dominated by obscure plant compounds and traditional Chinese medicine ingredients. Examples:
- Migraine #1 = corilagin (obscure tannin)
- Tuberculosis #1 = ME1036 (experimental antibiotic)
- Lyme #1 = Shuangshen (TCM formula)
- **Corilagin appeared at #1 for Migraine, Fibromyalgia, AND Amnesia** — same obscure compound dominating completely unrelated diseases

**Why:** marker/mechanism is dominated by the *same heavily-studied common drugs* (cyclosporine, methotrexate, etc.) as the therapeutic class. By using only m/m as negatives, the model learned "common, well-studied drugs = NEGATIVE for many diseases" and pushed them all out of every top-20. Obscure compounds with no m/m record floated up.

### What mixing does

50/50 and 25/75 mixes recover most baseline wins while still demoting the most obvious harmful drugs (cocaine for alcoholism, scopolamine for amnesia). But:
- **Lyme #1 went from Cyclosporine (baseline) → Prednisone (25% mix) → Rifampin (50% mix).** 50% finally gets a real treatment, but at the cost of more degraded wins.
- **Migraine baseline (Topiramate #1) is partially lost** — even 25% mix demotes it to "Gabapentin #1, Topiramate #?".

### The real lesson

**Hard-negative training has an irreducible trade-off:** more hard negatives → better safety on documented failures → degraded ranking on documented wins. There is no "free lunch" mix.

Honest implication for the paper: hard negatives **complement but cannot replace** explicit harm filtering. The Perplexity structured-judge approach (Days 8-9 in the plan) directly addresses harm without disrupting the embedding-similarity rankings the model produces. They should be **layered, not substituted**.

### Decision

- **Do NOT ship a retrained model as production yet.** The trade-offs aren't acceptable for the user-facing app.
- **Keep all variants in the repo** for the paper's experimental section.
- **Move on to Perplexity structured judge (Days 8-9)** — that's the real safety mechanism.
- **Revisit hard-negative training as a *complement* to Perplexity** in the final paper write-up.

### Files

- [scripts/retrain_with_hard_negatives.py](../scripts/retrain_with_hard_negatives.py) — `--hard-negative-fraction 0.0..1.0`, `--label NAME`
- [scripts/diff_regression.py](../scripts/diff_regression.py) — side-by-side comparison with positive/negative drug watchlists per disease
- `models/biolink_v2_hardneg.pt` (100%), `biolink_v2_hardneg-50.pt` (50%), `biolink_v2_hardneg-25.pt` (25%) — all variants saved
- `data/temperature_hardneg*.json`, `data/val_logits_hardneg*.npy` — corresponding calibration

## Candidate-count finding (2026-05-01)

After applying prior correction, the actual count of high-probability candidates per disease varies wildly. Top-20 display is hiding meaningful signal in some cases and overstating it in others.

| Disease | Strong (≥0.30) | Moderate (≥0.10) | Total |
|---|---|---|---|
| Fibromyalgia | **4** | 29 | 7,164 |
| Lyme Disease | 10 | 25 | 7,164 |
| Tuberculosis | 26 | 81 | 7,164 |
| Diabetes T2 | 30 | 185 | 7,164 |
| Alcoholism | 49 | 144 | 7,164 |
| Migraine | 56 | 200 | 7,164 |
| Dermatitis | 58 | 237 | 7,164 |
| Asthma | 69 | 249 | 7,164 |
| RA | 78 | 164 | 7,164 |
| **Amnesia** | **951** | 2,516 | 7,164 |

**Implications:**

1. **Top-20 hides real candidates** for diseases with rich therapeutic literature. Dermatitis (58 Strong), Asthma (69), RA (78) — user sees less than half. Someone wanting a less-common but plausible eczema treatment never sees ranks 21–58.

2. **Top-20 misleads for thin-signal diseases.** Fibromyalgia has only 4 Strong candidates. The UI shows 20 results with similar visual weight, hiding that ranks 5–20 are actually Moderate/Speculative.

3. **Amnesia is catastrophically broken.** 951 of 7,164 drugs (13%) score as Strong candidates. The model's embedding for Amnesia is too close to too many drug embeddings — fundamental cluster mismatch. Hard-negative training will help but won't fully fix this.

### UI fix (added to plan: P1, days 9-10)

**Combined approach (option C + B):**
- **Tier breakdown header above results:** "4 Strong, 29 Moderate, 7,131 Speculative — showing top 20"
- **"Show more" affordance below top-20:** if more Strong/Moderate candidates exist beyond the top 20, surface them in a "Show next 20 candidates" expandable section. Cap at 100 to avoid Amnesia-style 951-result lists.

**Files to touch:**
- `ui/components.py` — add tier breakdown header rendering
- `app.py` — add "show more" state management; add tier counts to inference response
- `core/inference.py` — return tier counts (Strong/Moderate/Speculative across all 7163) alongside top-N results, so UI doesn't need to recompute

This change is downstream of recalibration — the tier counts are only meaningful with prior correction applied.

## Wins documented (use as regression tests — these must NOT degrade)

| Disease | What's there | Why it matters |
|---|---|---|
| Fibromyalgia | Pregabalin #2, Milnacipran #3, Duloxetine #9 (all FDA-approved, all repurposed) + Naltrexone #6 (emerging) | Best aha + emerging discovery example |
| Dermatitis | Pimecrolimus #1, Tacrolimus #8, Cyclosporine #9 (transplant drugs as eczema cream) | Universal recognition |
| Migraine | Topiramate #1 (epilepsy), Amitriptyline #5 (antidepressant) | Two clean repurposings, both Standard of Care |
| RA, TB, Asthma | Mostly correct top 20 | Baselines showing the model works when cluster matches |
| Alopecia | Methotrexate #2 (autoimmune hair loss) | Real off-label use |

## Root-cause ML mistakes

### P0 — Random negative sampling
**File:** `v1_source/biolink-ctd-drug-disease-main/src/data_loader.py:84-93`
Negatives are random `(drug, disease)` pairs. CTD `marker/mechanism` evidence (causal/harmful pairs) is filtered out of positives but never added as negatives. Model never learns what "harmful" looks like.

**Fix (revised):** Sample hard negatives from CTD `marker/mechanism` rows — **but filter by evidence strength**, not bulk. Only use rows where:
- CTD `InferenceScore` is high, OR
- Multiple supporting publications/genes exist, OR
- The wording explicitly indicates causation (not just biomarker correlation)

This shrinks the negative pool but ensures every negative is genuinely "studied and known to be problematic" rather than "happened to co-occur in one paper." Avoids killing under-studied genuine candidates.

### P0 — Pair-level train/test split
**File:** `v1_source/biolink-ctd-drug-disease-main/src/data_loader.py:127-141`
Splits by pair, not entity. AUC=0.947 is inflated.

**Fix:** Cold-start splits — hold out entire drugs OR entire diseases. Report both AUCs. Cold-start will be much lower; that's the honest number.

### P1 — Frozen mean-pooled BioWordVec embeddings
**File:** `v1_source/biolink-ctd-drug-disease-main/src/embeddings.py:46-50`
Mean-pooling tokens loses composition. BioWordVec is co-occurrence only — no relational structure. Frozen, so MLP can't fix it.

**Fix (cheap):** swap to a sentence transformer (already imported in `embeddings.py` but not used).
**Fix (real):** fine-tune embeddings, or use a knowledge-graph embedding (TransE, ComplEx) on DrugBank/CTD relations. Out of 2-week scope.

### P1 — Symmetric pair features
**File:** `v1_source/biolink-ctd-drug-disease-main/src/embeddings.py:119-121`
`[a, b, |a-b|, a*b]` is symmetric — model has no concept of "treats" as a directed relation.

**Fix:** asymmetric scoring (bilinear layer, or relational scoring). Out of 2-week scope.

### P1 — Structured Perplexity re-ranking (NOT regex-based)
**File:** new — would live in `core/inference.py` after enrichment runs.

**Original idea was regex-matching free-text Perplexity output for "harmful", "contraindicated", etc.** That's fragile — would catch general drug toxicity ("cocaine is harmful") regardless of whether the harm relates to the queried indication.

**Fix (revised):** Send Perplexity a structured query for each top candidate:

> *"For the use of {drug} as a treatment for {disease}: (1) Are clinical trials currently investigating this combination? (2) Is there evidence of harm or contraindication when {drug} is used specifically for {disease}? (3) Single verdict: STANDARD / PROMISING / UNKNOWN / HARMFUL."*

Then re-rank on the structured verdict:

```
× 1.5 boost     if verdict = STANDARD (already validated treatment)
× 1.2 boost     if verdict = UNKNOWN AND active trials exist (emerging — discovery boost)
× 1.0 unchanged if verdict = UNKNOWN with no trial activity (pure discovery space — preserve)
× 1.0 unchanged if verdict = PROMISING (preliminary signal — preserve)
× 0.3 demote    if verdict = HARMFUL (specifically for this indication)
```

**Critical:** the harm must be tied to the specific drug-disease pairing, not general drug toxicity. The structured verdict format enforces that.

### P2 — Score saturation / prior mismatch
**File:** `v1_source/biolink-ctd-drug-disease-main/src/models.py:76`
1:1 positive:negative training ratio doesn't match inference reality (~0.1% prior). Every top-20 saturates at 97-100% Strong.

**Fix:** train with 1:50 or 1:100 negative ratio, or focal loss, or recalibrate post-hoc against the inference distribution.

### P2 — No baseline in evaluation
**File:** `v1_source/biolink-ctd-drug-disease-main/src/evaluation.py`
AUC reported with no comparison to (a) random, (b) drug popularity, (c) plain BioWordVec cosine similarity.

**Fix:** add baselines. If MLP barely beats cosine, that's a finding worth reporting.

## Regression test suite (mandatory validation gate)

Before AND after every fix, run all 12 queries below and document top 20 + verdicts. Any fix that degrades a win is rejected or refined.

**Wins to preserve (must NOT degrade):**
1. Fibromyalgia — Pregabalin/Milnacipran/Duloxetine in top 10
2. Dermatitis — Pimecrolimus/Tacrolimus/Cyclosporine in top 10
3. Migraine — Topiramate #1, Amitriptyline #5
4. Rheumatoid Arthritis — DMARDs/corticosteroids dominant
5. Tuberculosis — Isoniazid/Rifampin/Ethambutol/Pyrazinamide in top 5
6. Asthma — Leukotriene antagonists + ICS dominant

**Failures to improve:**
7. Lyme — cyclosporine/methylpred should drop out of top 10; doxycycline/azithromycin should rise
8. Type 2 Diabetes — streptozocin and nicotine should drop out of top 20
9. Alcoholism — cocaine/heroin/PCP should drop; naltrexone/baclofen should rise
10. Insomnia — nicotine should drop; trazodone/melatonin should rise
11. ADHD — methylphenidate should rise into top 5; CBD/scopolamine should drop
12. Amnesia — scopolamine/strychnine/PCP should drop; donepezil/galantamine should rise

**Emerging to actively surface (discovery boost gate):**
- Naltrexone for Fibromyalgia (currently #6) — should move up if discovery boost works correctly

## Revised 2-week plan

| Days | Task | Validation gate | Deploy? |
|---|---|---|---|
| ~~1~~ ✅ | Lock regression test suite — runner + baseline JSON saved | Done 2026-05-01 | No |
| ~~2-3~~ ✅ | Baselines (popularity, cosine, random) on all 2526 diseases | Done 2026-05-01 — popularity at 0.88, MLP +6.7pts | No |
| ~~4-7~~ ⚠️ | Hard-negative retraining (3 variants tried) | Done 2026-05-01 — found irreducible trade-off, NOT shipping. See "Hard-negative trade-off" section. | No (per finding) |
| ~~8-9 (judge)~~ ✅ | Structured Perplexity judge for harm | Done 2026-05-01 — `HARM_FOR_INDICATION` field added; harm-only re-ranking in core/reranking.py; 21 tests | Pending UI tier counts before deploy |
| **8-9 (UI)** | UI tier counts above results + "show more" affordance | Manual UI test in Streamlit | **Yes — bundle with reranking deploy** |
| **10** | Discovery boost for UNKNOWN verdict + active trials + "show more" affordance | Naltrexone-fibro should move up; user can browse beyond top 20 | **Yes — Day 10** (inference.py + app.py changes) |
| ~~11~~ ✅ | Inference-prior recalibration (score spread fix) | Done 2026-05-01 — moved up because impact was bigger than expected | Pending deploy with next wave |
| **12-14** | Write up — paper limitations explicit, not hidden | — | Final deploy + tag release |

## Deployment workflow (after each "Yes" milestone above)

The HF Space rebuilds on push. The Dockerfile copies everything, so updated weights / cached embeddings / calibration / code all ship by being in the git push.

```bash
# 1. Local validation first
streamlit run app.py            # smoke-test locally
python scripts/run_regression.py # confirm regression suite passes

# 2. Commit
git add -A
git commit -m "feat: hard-negative retraining (improves direction blindness)"

# 3. Push to both remotes
git push origin main            # GitHub source of truth
git push hf main                # Triggers HF Space rebuild (~3-5 min)

# 4. Verify deployed app
# Open https://huggingface.co/spaces/dustinhaggett/biolink-v2
# Run 2-3 regression queries through the UI to confirm
```

**File size watch:** HF git repos have a 10 MB per-file soft limit (larger needs git-lfs). Check before each deploy:
- `models/biolink_v1.pt` — 810 KB ✅
- `data/drug_embeddings.npy` (7163 × 200 × 4 bytes) — ~5.7 MB ✅
- `data/disease_embeddings.npy` (2525 × 200 × 4 bytes) — ~2 MB ✅
- If we ever swap to a sentence transformer (768-dim instead of 200), drug embeddings go to ~22 MB → needs git-lfs

**Deployment-specific risks:**
1. **HF Space cold-start** — first request after redeploy is slow (loads model + embeddings). Document this; don't expect instant queries on a freshly-rebuilt Space.
2. **Cached embeddings stale after retraining** — if hard-negative training changes the embedding scheme (it shouldn't, since BioWordVec stays frozen), the cached `.npy` files would need regeneration. Same for any sentence-transformer swap.
3. **Perplexity API key in HF Space settings** — if the structured judge prompt changes, no env var changes needed. If we add a new API (e.g., a separate harm-classification model), update HF Space "Settings → Variables and secrets" before pushing.
4. **App.py changes** — if we add UI to expose the discovery boost separately ("Established treatments" / "Emerging candidates" sections), test in local Streamlit first; HF Space takes 3-5 min to rebuild and surfacing UI bugs there is slow.

## Pre-flight checklist for the final push (Day 14)

- [ ] All 12 regression queries pass (failures improved, wins preserved)
- [ ] Naltrexone-fibro discovery boost verified
- [ ] Score distribution no longer ceilings at 99-100%
- [ ] Cold-start AUC documented in paper
- [ ] Baseline comparisons (cosine, popularity) documented
- [ ] Limitations section written
- [ ] HF Space rebuilt and smoke-tested through the UI
- [ ] Tag a git release (`v2.1` or similar) on `origin`
- [ ] **Rotate the HF write token** (currently exposed in `.git/config`)

## What's different from the original plan

1. **Hard negatives are evidence-filtered, not bulk** — protects under-studied genuine candidates from being labeled false negatives
2. **Re-ranking uses structured LLM verdict, not regex** — harm specifically tied to indication, not general toxicity
3. **Active discovery boost added** — emerging candidates with trial activity surface higher
4. **Mandatory regression test gate** — every fix tested against documented wins AND failures
5. **Limitations documented explicitly in the paper** — not hidden

## Known limits the plan does NOT fix

Be explicit about these in the paper. Pretending we fixed everything is worse than honest scoping.

1. **Discovery floor set by CTD curation bias.** CTD heavily favors pharmaceutical-industry research. Underrepresents traditional medicine, herbal compounds, food-derived molecules, off-patent drugs. The model will always be biased toward "pharmaceutical-shaped" candidates.

2. **No mechanism-based reasoning.** Pipeline treats drugs/diseases as text, not biology. Cannot make true first-degree discoveries (drug-target → disease-pathway). Best suited for second-degree repurposing within established pharmacological neighborhoods.

3. **Cluster mismatch (Lyme, HD).** Even with all fixes, the model will still favor drugs popular in the closest embedding cluster. Real fix requires KG embeddings — out of 2-week scope.

4. **Trial-outcome blindness (ALS).** Model rewards research interest regardless of trial outcomes. Real fix requires ClinicalTrials.gov outcome features as model inputs.

5. **Perplexity judge has its own biases.** Conservative, mainstream-leaning. Will mark genuinely novel ideas as "no evidence" — but the discovery boost step partially mitigates by treating "UNKNOWN + active trials" as a positive signal.

## What goes in the final paper

**Limitations section:** Full failure catalog with screenshots. Document the documented failures and what fixes did/didn't address them.

**Discovery vs. harm framing:** Explicit contribution. *"A drug repurposing tool surfaces novel candidates by definition; valuable hits often lack established evidence. The challenge is distinguishing 'unstudied' from 'studied and harmful for this indication.' Our fixes target the latter without filtering the former."*

**Honest claim:** The Perplexity evidence layer is doing safety work the MLP cannot do alone. That's a deliberate architectural choice. Hard-negative training improves direction-awareness within the model itself; structured Perplexity re-ranking catches what slips through.

**Bounded scope:** *"Best suited for repurposing within established pharmacological neighborhoods. Cannot identify first-degree novel discoveries requiring mechanistic reasoning (drug-target → disease-pathway). Discovery floor is set by CTD curation bias toward pharmaceutical-industry-funded research."*

**What we'd build with more time:** Knowledge-graph embeddings (DrugBank + CTD relations), multi-class evidence labeling (therapeutic / causal / marker / unstudied), ClinicalTrials.gov outcome features, mechanism-aware features (drug targets, disease pathways).
