"""
Harm-aware re-ranking of model predictions using Perplexity evidence verdicts.

Per the post-presentation design (docs/POST_PRESENTATION_TODO.md), the model's
ranking is adjusted using structured Perplexity verdicts BUT only to demote
candidates with evidence of indication-specific harm. Candidates with no
evidence (the discovery space) are preserved.

Design rules — DO NOT relax these without revisiting the discovery-vs-harm
tension explicitly:

  1. NEVER demote on absence of evidence. "Insufficient evidence" / "Unknown"
     are EXACTLY the candidates a discovery tool should surface.

  2. Demote ONLY on evidence of harm for THIS indication. General drug toxicity
     (e.g., "cocaine is dangerous") should not demote cocaine for cancer if
     there's no evidence specifically about cocaine for cancer being harmful.

  3. Boost on validated therapeutic evidence — these are candidates the user
     is most likely to trust and act on first.

  4. Active clinical trial signal can mildly boost UNKNOWN candidates (the
     "naltrexone-for-fibromyalgia" pattern — drugs being investigated even
     without an established verdict).

Multipliers are applied to calibrated probabilities, then results are
re-sorted. Ranking is recomputed; original logits are preserved for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# Verdict labels emitted by enrichment.perplexity._parse_verdict() (lowercased).
VERDICT_SUPPORTS = "supports"
VERDICT_STANDARD = "standard-of-care"
VERDICT_CONFLICTS = "conflicts"
VERDICT_INSUFFICIENT = "insufficient"

# Indication-specific harm flag emitted by enrichment.perplexity._parse_harm()
HARM_HARMFUL = "harmful"
HARM_NOT_HARMFUL = "not_harmful"
HARM_UNKNOWN = "unknown"


@dataclass(frozen=True)
class RerankConfig:
    """Multiplier table for evidence-based re-ranking.

    Defaults are the result of the discovery-vs-harm reasoning in the post-
    presentation TODO. Tuning should be evidence-driven (regression suite),
    not vibes.
    """

    boost_standard_of_care: float = 1.5      # Established clinical practice
    boost_supports: float = 1.3              # Strong supporting evidence
    boost_unknown_with_trials: float = 1.15  # Discovery candidate with active trials
    no_change: float = 1.0
    demote_unvalidated_interactions: float = 0.5  # Interactions flagged + no positive validation
    demote_harmful: float = 0.3              # Evidence of indication-specific harm
    demote_harmful_with_interactions: float = 0.1  # Harm + drug interactions = stay-away


def _has_active_trials(result: dict) -> bool:
    """True if the enrichment layer found ≥1 ACTIVE/RECRUITING trial.

    Strict check — completed trials don't qualify because they tell us about
    the past, not ongoing investigation. Trials with unknown status count as
    "active" (we'd rather over-surface a discovery candidate than under-surface).
    """
    trials = result.get("clinical_trials") or []
    if not trials:
        return False
    for t in trials:
        status = (t.get("status") or "").upper()
        if status in {"RECRUITING", "ACTIVE", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING", "UNKNOWN", ""}:
            return True
    return False


def _multiplier(result: dict, config: RerankConfig) -> tuple[float, str]:
    """Return (multiplier, reason) for a single result based on its evidence.

    Order of precedence (later rules cannot override earlier ones):
      1. Indication-specific harm (always demote, regardless of other signals)
      2. Standard-of-care or supports verdict (boost)
      3. Drug interactions flagged AND no positive validation (mild demote)
      4. Insufficient/unknown + active trial AND no interactions (discovery boost)
      5. Default (no change)
    """
    evidence = result.get("evidence") or {}
    verdict = (evidence.get("verdict") or "").lower()
    harm = (evidence.get("harm_for_indication") or "").lower()
    has_interactions = bool(evidence.get("has_interactions", False))

    # Rule 1: HARM trumps everything. Indication-specific harm is unambiguous.
    if harm == HARM_HARMFUL:
        if has_interactions:
            return config.demote_harmful_with_interactions, "harm + drug interactions"
        return config.demote_harmful, "indication-specific harm"

    # Rule 2: Validated therapeutic evidence → boost. SoC/SUPPORTS positive
    # validation overrides any drug-interactions flag — well-established
    # treatments often interact with other meds; that's a prescribing concern,
    # not a "this drug shouldn't be on the list" concern.
    if verdict == VERDICT_STANDARD:
        return config.boost_standard_of_care, "standard of care"
    if verdict == VERDICT_SUPPORTS:
        return config.boost_supports, "evidence supports"

    # Rule 3: Mild demote for "interactions flagged but no positive validation."
    # Catches the Amphetamine-for-Alcoholism pattern: Perplexity classified harm
    # as INSUFFICIENT (its harm field is sensitive to phrasing) but correctly
    # flagged drug interactions. Without an active boost from SoC/SUPPORTS, we
    # treat the interactions flag as a mild safety signal — not as definitive
    # harm (which would be × 0.3) but as "this is probably not a candidate the
    # user should see in the top 5."
    if has_interactions:
        return config.demote_unvalidated_interactions, "drug interactions, unvalidated"

    # Rule 4: Discovery boost for unstudied candidates with active trial activity.
    # Important guard: skip if has_interactions=True (handled above). Without
    # this guard, drugs like methamphetamine for alcoholism would get boosted
    # because Perplexity sometimes classifies clearly-dangerous combos as
    # INSUFFICIENT-not-HARMFUL while correctly flagging interactions. The
    # interactions flag is a second layer of safety for when the primary harm
    # classification misses. See docs/POST_PRESENTATION_TODO.md.
    if (
        verdict in (VERDICT_INSUFFICIENT, "", "unknown")
        and _has_active_trials(result)
    ):
        return config.boost_unknown_with_trials, "active trials, no established verdict"

    # Default — preserve. CONFLICTS without harm/interactions is preserved
    # (mixed evidence but no specific contraindication for THIS indication).
    # INSUFFICIENT without active trials is preserved (pure discovery space).
    return config.no_change, "no adjustment"


def apply_evidence_reranking(
    results: list[dict],
    config: RerankConfig | None = None,
) -> list[dict]:
    """Re-rank results in place based on Perplexity evidence verdicts.

    Args:
        results: List of result dicts. Each must have at least 'drug' and 'proba'.
                 Expected to have 'evidence' from search_drug_disease() for the
                 entries that were enriched (typically only the top-N).
                 Entries WITHOUT 'evidence' are treated as no_change (default
                 multiplier) — they keep their original score.
        config:  RerankConfig, or None for defaults.

    Returns:
        New sorted list (descending by adjusted_proba). Each result gains:
          - 'reranked_proba': float — adjusted probability (same as 'proba' after rerank)
          - 'rerank_multiplier': float — what was applied
          - 'rerank_reason': str — human-readable explanation
          - 'original_rank': int — pre-rerank position (1-indexed)
          - 'model_proba': float — pre-rerank probability (model only, no evidence)
          - 'model_tier': str — pre-rerank tier
        The user-facing 'proba', 'tier', and 'rank' are OVERWRITTEN to reflect the
        rerank — so UI components that read 'proba'/'tier' show the evidence-adjusted
        view by default. Originals are preserved as 'model_proba'/'model_tier' for
        debugging or "show details" UI.

    Properties:
      - Pure function. Input results are NOT mutated; new list returned.
      - Order-preserving for ties (Python's sort is stable).
      - Safe on missing fields. Defaults gracefully if 'evidence' absent.
    """
    # Local import to avoid circular dep with core.calibration
    from core.calibration import confidence_tier

    config = config or RerankConfig()

    # Make a defensive copy so callers don't see mutation
    new_results = []
    for i, r in enumerate(results, start=1):
        new = dict(r)
        new["original_rank"] = i
        mult, reason = _multiplier(new, config)
        new["rerank_multiplier"] = mult
        new["rerank_reason"] = reason

        # Preserve the model-only confidence under separate keys for transparency
        original_proba = float(new.get("proba", 0.0))
        new["model_proba"] = original_proba
        new["model_tier"] = new.get("tier", confidence_tier(original_proba))

        # Overwrite user-facing fields so UI shows the evidence-adjusted view.
        # This was a 2026-05-01 bugfix: previously the reranker added reranked_proba
        # but didn't update proba/tier, so the UI showed pre-rerank values next to
        # post-rerank rank positions — e.g. "Nicotine #20: 69% Strong" with a HARMFUL
        # badge, even though the rerank had multiplied the score by 0.1.
        adjusted = max(0.0, min(1.0, original_proba * mult))
        new["reranked_proba"] = adjusted
        new["proba"] = adjusted
        new["tier"] = confidence_tier(adjusted)

        new_results.append(new)

    # Sort by adjusted probability, descending. Stable so ties keep input order.
    new_results.sort(key=lambda r: r["reranked_proba"], reverse=True)

    # Rewrite rank to reflect new position
    for new_rank, r in enumerate(new_results, start=1):
        r["rank"] = new_rank

    return new_results


def explain_changes(before: list[dict], after: list[dict], top_n: int = 10) -> Iterable[str]:
    """Yield human-readable lines describing rerank effects on the top-N.

    Used for diagnostic output and the regression-suite logs. Compares ranks
    before/after rerank.
    """
    after_by_drug = {r["drug"]: r for r in after}
    for r in before[:top_n]:
        drug = r["drug"]
        new = after_by_drug.get(drug)
        if not new:
            continue
        old_rank = r.get("rank", "?")
        new_rank = new.get("rank", "?")
        if old_rank == new_rank:
            arrow = "="
        elif isinstance(old_rank, int) and isinstance(new_rank, int):
            arrow = f"↑{old_rank - new_rank}" if new_rank < old_rank else f"↓{new_rank - old_rank}"
        else:
            arrow = "?"
        reason = new.get("rerank_reason", "?")
        yield f"  #{old_rank:>2} → #{new_rank:<2} [{arrow:>4}]  {drug:<35} ({reason})"
