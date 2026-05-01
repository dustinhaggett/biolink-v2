"""
Tests for harm-aware re-ranking.

These are pure-function tests with hand-crafted evidence dicts — no Perplexity
API calls, no model loading. They lock in the discovery-vs-harm design
described in core/reranking.py and docs/POST_PRESENTATION_TODO.md.
"""

from __future__ import annotations

import pytest

from core.reranking import (
    HARM_HARMFUL,
    HARM_NOT_HARMFUL,
    HARM_UNKNOWN,
    VERDICT_CONFLICTS,
    VERDICT_INSUFFICIENT,
    VERDICT_STANDARD,
    VERDICT_SUPPORTS,
    RerankConfig,
    apply_evidence_reranking,
)


def _result(
    drug: str,
    proba: float,
    rank: int,
    verdict: str | None = None,
    harm: str | None = None,
    interactions: bool = False,
    trials: list[dict] | None = None,
) -> dict:
    """Helper to build a result dict with optional evidence."""
    r = {"drug": drug, "proba": proba, "rank": rank}
    if verdict is not None or harm is not None or interactions or trials is not None:
        r["evidence"] = {}
        if verdict is not None:
            r["evidence"]["verdict"] = verdict
        if harm is not None:
            r["evidence"]["harm_for_indication"] = harm
        r["evidence"]["has_interactions"] = interactions
        if trials is not None:
            r["clinical_trials"] = trials
    return r


# ---------------------------------------------------------------------------
# Critical invariants from the discovery-vs-harm design
# ---------------------------------------------------------------------------

class TestDoNotDemoteOnAbsenceOfEvidence:
    """The single most important property — preserves novel candidates."""

    def test_insufficient_evidence_unknown_harm_is_unchanged(self):
        """Pure discovery space — must not be touched."""
        results = [_result("MysteryDrugX", 0.50, 1, verdict=VERDICT_INSUFFICIENT, harm=HARM_UNKNOWN)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.50)
        assert out[0]["rerank_multiplier"] == 1.0

    def test_no_evidence_dict_is_unchanged(self):
        """Result with no 'evidence' key (e.g., not enriched) must not be touched."""
        results = [_result("UnenrichedDrug", 0.42, 1)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.42)
        assert out[0]["rerank_multiplier"] == 1.0

    def test_conflicts_without_harm_is_preserved(self):
        """Mixed evidence ≠ harm. Don't filter out diverging-results candidates."""
        results = [_result("MixedEvidenceDrug", 0.30, 1, verdict=VERDICT_CONFLICTS, harm=HARM_UNKNOWN)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.30)

    def test_conflicts_with_not_harmful_is_preserved(self):
        """Conflicts but explicitly not harmful for this indication → preserve."""
        results = [_result("DebatedDrug", 0.40, 1, verdict=VERDICT_CONFLICTS, harm=HARM_NOT_HARMFUL)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.40)


class TestHarmDemotion:
    """Demote ONLY on indication-specific harm."""

    def test_indication_harmful_demotes_to_03x(self):
        results = [_result("CocaineForAlcoholism", 0.90, 1, harm=HARM_HARMFUL)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.90 * 0.3)
        assert "harm" in out[0]["rerank_reason"].lower()

    def test_harm_with_interactions_demotes_to_01x(self):
        """Stay-away level — known harm AND drug interactions."""
        results = [_result("DangerousCombo", 0.80, 1, harm=HARM_HARMFUL, interactions=True)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.80 * 0.1)

    def test_harm_overrides_supports_verdict(self):
        """If a drug 'supports' some general use but is harmful for THIS indication,
        the harm signal wins. (Edge case but the rule is harm-trumps-everything.)"""
        results = [_result("WeirdDrug", 0.70, 1, verdict=VERDICT_SUPPORTS, harm=HARM_HARMFUL)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.70 * 0.3)


class TestEvidenceBoosts:
    def test_standard_of_care_boosts_15x(self):
        results = [_result("Naltrexone", 0.40, 1, verdict=VERDICT_STANDARD, harm=HARM_NOT_HARMFUL)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.40 * 1.5)

    def test_supports_boosts_13x(self):
        results = [_result("PromisingDrug", 0.30, 1, verdict=VERDICT_SUPPORTS, harm=HARM_NOT_HARMFUL)]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.30 * 1.3)

    def test_proba_capped_at_1_0_when_boosted(self):
        results = [_result("AlreadyHigh", 0.80, 1, verdict=VERDICT_STANDARD)]
        out = apply_evidence_reranking(results)
        # 0.80 * 1.5 = 1.2 → must clamp to 1.0
        assert out[0]["reranked_proba"] == pytest.approx(1.0)


class TestDiscoveryBoost:
    def test_unknown_with_recruiting_trial_gets_mild_boost(self):
        """Naltrexone-for-fibromyalgia pattern — emerging candidate with trials."""
        results = [_result(
            "EmergingCandidate", 0.20, 1,
            verdict=VERDICT_INSUFFICIENT, harm=HARM_UNKNOWN,
            trials=[{"id": "NCT123", "status": "RECRUITING"}],
        )]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.20 * 1.15)
        assert "trials" in out[0]["rerank_reason"].lower()

    def test_unknown_without_trials_is_unchanged(self):
        results = [_result(
            "UnstudiedDrug", 0.20, 1,
            verdict=VERDICT_INSUFFICIENT, harm=HARM_UNKNOWN, trials=[],
        )]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.20)

    def test_completed_trials_alone_dont_trigger_discovery_boost(self):
        """Old completed trials = past, not active investigation."""
        results = [_result(
            "OldDrug", 0.20, 1,
            verdict=VERDICT_INSUFFICIENT, harm=HARM_UNKNOWN,
            trials=[{"id": "NCT001", "status": "COMPLETED"}],
        )]
        out = apply_evidence_reranking(results)
        assert out[0]["reranked_proba"] == pytest.approx(0.20)

    def test_drug_interactions_blocks_discovery_boost(self):
        """Methamphetamine-for-alcoholism pattern: Perplexity classifies harm
        as UNKNOWN (the field is sensitive to phrasing) but correctly flags
        Drug Interactions. Without this guard, the discovery boost fires and
        promotes a clearly-dangerous combo. Interactions=True is a backstop.

        Note: Rule 3 (unvalidated-interactions demote) now also fires here, so
        the result is actively demoted to ×0.5 — even better than just blocking
        the boost. This test asserts the discovery-boost reason isn't applied."""
        results = [_result(
            "MethForAlcoholism", 0.93, 1,
            verdict=VERDICT_INSUFFICIENT, harm=HARM_UNKNOWN,
            interactions=True,
            trials=[{"id": "NCT_ACTIVE", "status": "RECRUITING"}],
        )]
        out = apply_evidence_reranking(results)
        # Demoted to ×0.5 by Rule 3, NOT boosted ×1.15 by Rule 4
        assert out[0]["reranked_proba"] == pytest.approx(0.93 * 0.5)
        assert "active trials" not in out[0]["rerank_reason"].lower()
        assert "interactions" in out[0]["rerank_reason"].lower()


class TestUnvalidatedInteractionsDemote:
    """Rule 3: drug interactions flagged + no SoC/SUPPORTS validation = mild demote.

    Catches the Amphetamine-for-Alcoholism case: verdict=INSUFFICIENT,
    interactions=True, harm=UNKNOWN. Without this rule, the candidate sits at
    its model-only score (75%+) and confuses users about safety.
    """

    def test_insufficient_plus_interactions_demotes_to_05x(self):
        result = _result("Amphetamine", 0.75, 1,
                         verdict=VERDICT_INSUFFICIENT, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.75 * 0.5)
        assert "interactions" in out[0]["rerank_reason"].lower()

    def test_conflicts_plus_interactions_demotes(self):
        """Mixed evidence + dangerous interaction = demote."""
        result = _result("X", 0.70, 1, verdict=VERDICT_CONFLICTS, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.70 * 0.5)

    def test_unknown_plus_interactions_demotes(self):
        result = _result("X", 0.50, 1, harm=HARM_UNKNOWN, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.50 * 0.5)

    def test_standard_of_care_plus_interactions_still_boosts(self):
        """SoC validation overrides interactions warning. Established treatments
        often have interactions; that's a prescribing concern, not a 'don't show
        this in top 5' concern. Chlordiazepoxide for Alcoholism is the exemplar:
        SoC + interactions, should still rank top."""
        result = _result("Chlordiazepoxide", 0.50, 1,
                         verdict=VERDICT_STANDARD, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.50 * 1.5)
        assert "standard" in out[0]["rerank_reason"].lower()

    def test_supports_plus_interactions_still_boosts(self):
        result = _result("X", 0.40, 1, verdict=VERDICT_SUPPORTS, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.40 * 1.3)

    def test_interactions_alone_no_verdict_demotes(self):
        """Edge case: interactions flagged with no verdict at all."""
        result = _result("X", 0.50, 1, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.50 * 0.5)

    def test_no_interactions_no_demote(self):
        """Default behavior: no interactions, no demote."""
        result = _result("X", 0.50, 1, verdict=VERDICT_CONFLICTS)
        out = apply_evidence_reranking([result])
        assert out[0]["reranked_proba"] == pytest.approx(0.50)  # × 1.0


# ---------------------------------------------------------------------------
# Re-ranking behavior (sort, ranks, original_rank tracking)
# ---------------------------------------------------------------------------

class TestRerankingBehavior:
    def test_sort_descending_by_adjusted_proba(self):
        results = [
            _result("CocaineHarmful",   0.95, 1, harm=HARM_HARMFUL),       # 0.95 * 0.3 = 0.285
            _result("NaltrexoneStd",    0.40, 2, verdict=VERDICT_STANDARD), # 0.40 * 1.5 = 0.60
            _result("UnknownNeutral",   0.50, 3),                          # 0.50 * 1.0 = 0.50
        ]
        out = apply_evidence_reranking(results)
        assert [r["drug"] for r in out] == ["NaltrexoneStd", "UnknownNeutral", "CocaineHarmful"]
        assert [r["rank"] for r in out] == [1, 2, 3]

    def test_original_rank_preserved(self):
        results = [
            _result("A", 0.95, 1, harm=HARM_HARMFUL),
            _result("B", 0.40, 2, verdict=VERDICT_STANDARD),
        ]
        out = apply_evidence_reranking(results)
        # Find each drug in output, check original_rank
        d = {r["drug"]: r for r in out}
        assert d["A"]["original_rank"] == 1
        assert d["B"]["original_rank"] == 2

    def test_input_not_mutated(self):
        """Defensive: input results dict must not change."""
        original = _result("X", 0.5, 1, verdict=VERDICT_STANDARD)
        original_copy = dict(original)
        original_copy["evidence"] = dict(original["evidence"])
        apply_evidence_reranking([original])
        # Top-level keys unchanged
        for key in ["drug", "proba", "rank"]:
            assert original[key] == original_copy[key]
        # Should NOT have rerank fields added
        assert "reranked_proba" not in original
        assert "rerank_multiplier" not in original

    def test_empty_input_returns_empty(self):
        assert apply_evidence_reranking([]) == []

    def test_all_unenriched_preserves_order(self):
        """If nothing has evidence, ranking stays exactly as input."""
        results = [
            _result("A", 0.9, 1),
            _result("B", 0.7, 2),
            _result("C", 0.3, 3),
        ]
        out = apply_evidence_reranking(results)
        assert [r["drug"] for r in out] == ["A", "B", "C"]

    def test_only_top_n_enriched_preserves_unenriched_at_bottom(self):
        """Common case: top-5 enriched, ranks 6-20 not. Rerank works on subset."""
        results = [
            _result("Top1Harmful", 0.95, 1, harm=HARM_HARMFUL),    # → 0.285
            _result("Top2Strong",  0.92, 2, verdict=VERDICT_STANDARD),  # → 1.0 (clamped from 1.38)
            _result("NotEnriched3", 0.50, 3),  # → 0.50
            _result("NotEnriched4", 0.30, 4),  # → 0.30
        ]
        out = apply_evidence_reranking(results)
        # Order: Top2Strong (1.0), NotEnriched3 (0.50), NotEnriched4 (0.30), Top1Harmful (0.285)
        assert [r["drug"] for r in out] == ["Top2Strong", "NotEnriched3", "NotEnriched4", "Top1Harmful"]


class TestProbaAndTierOverwrite:
    """User-facing 'proba' and 'tier' must reflect post-rerank values, not pre-rerank.

    Bugfix 2026-05-01: previously the UI showed pre-rerank scores next to
    post-rerank ranks — e.g. Nicotine demoted to #20 but still displayed
    "69% Strong" with a HARMFUL badge.
    """

    def test_proba_overwritten_on_demotion(self):
        result = _result("Nicotine", 0.69, 1, harm=HARM_HARMFUL, interactions=True)
        out = apply_evidence_reranking([result])
        # Original was 0.69; harm + interactions = ×0.1 = 0.069
        assert out[0]["proba"] == pytest.approx(0.069, abs=1e-3)
        assert out[0]["tier"] == "Speculative"  # 0.069 < 0.10 threshold

    def test_proba_overwritten_on_boost(self):
        result = _result("Topiramate", 0.50, 1, verdict=VERDICT_STANDARD)
        out = apply_evidence_reranking([result])
        # Original 0.50; standard-of-care = ×1.5 = 0.75
        assert out[0]["proba"] == pytest.approx(0.75, abs=1e-3)
        assert out[0]["tier"] == "Strong"

    def test_proba_unchanged_on_no_adjustment(self):
        result = _result("Mystery", 0.40, 1)  # no evidence
        out = apply_evidence_reranking([result])
        assert out[0]["proba"] == pytest.approx(0.40)
        # tier may flip if 0.40 was below threshold but it's >= 0.30 = Strong
        assert out[0]["tier"] == "Strong"

    def test_model_proba_preserved_for_debugging(self):
        """Original score must be preserved as model_proba so transparency UI works."""
        result = _result("Nicotine", 0.69, 1, harm=HARM_HARMFUL, interactions=True)
        out = apply_evidence_reranking([result])
        assert out[0]["model_proba"] == pytest.approx(0.69)
        # Tier was Strong before rerank (0.69 >= 0.30); preserved
        assert out[0]["model_tier"] == "Strong"

    def test_reranked_proba_equals_proba_after_rerank(self):
        """Both fields should agree — proba IS the reranked value, reranked_proba kept for clarity."""
        result = _result("X", 0.40, 1, verdict=VERDICT_STANDARD)
        out = apply_evidence_reranking([result])
        assert out[0]["proba"] == out[0]["reranked_proba"]


# ---------------------------------------------------------------------------
# Real-world scenarios from documented failures
# ---------------------------------------------------------------------------

class TestDocumentedScenarios:
    """Simulate the failure cases from POST_PRESENTATION_TODO.md and verify
    that harm-aware reranking produces the expected behavior."""

    def test_alcoholism_demotes_cocaine_promotes_naltrexone(self):
        """Original baseline had Cocaine #2, Naltrexone #6.
        With evidence: Cocaine flagged HARMFUL, Naltrexone flagged STANDARD.
        Expectation: Naltrexone leapfrogs to top, Cocaine sinks."""
        results = [
            _result("Chlordiazepoxide", 0.999, 1, verdict=VERDICT_STANDARD),  # withdrawal med, real
            _result("Cocaine",         0.999, 2, harm=HARM_HARMFUL),
            _result("Heroin",          0.998, 3, harm=HARM_HARMFUL),
            _result("Phencyclidine",   0.998, 4, harm=HARM_HARMFUL),
            _result("Methamphetamine", 0.998, 5, harm=HARM_HARMFUL),
            _result("Naltrexone",      0.997, 6, verdict=VERDICT_STANDARD),
        ]
        out = apply_evidence_reranking(results)
        # The two STANDARD drugs (Chlordiazepoxide and Naltrexone) should be on top
        top2 = {r["drug"] for r in out[:2]}
        assert top2 == {"Chlordiazepoxide", "Naltrexone"}
        # Cocaine/Heroin/PCP/Meth should all be at the bottom
        bottom4 = {r["drug"] for r in out[-4:]}
        assert bottom4 == {"Cocaine", "Heroin", "Phencyclidine", "Methamphetamine"}

    def test_lyme_demotes_cyclosporine_when_flagged_harmful(self):
        """Lyme baseline had Cyclosporine #1 (immunosuppressant for bacterial infection).
        With evidence flagging it harmful + interactions → drops to bottom."""
        results = [
            _result("Cyclosporine",         0.99, 1, harm=HARM_HARMFUL, interactions=True),
            _result("Methylprednisolone",   0.99, 2, harm=HARM_HARMFUL, interactions=True),
            _result("Doxycycline",          0.99, 3, verdict=VERDICT_STANDARD),
        ]
        out = apply_evidence_reranking(results)
        assert out[0]["drug"] == "Doxycycline"
