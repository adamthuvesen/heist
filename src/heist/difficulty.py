"""Public, name-free difficulty weighting for the headline metric, alpha (α).

alpha is the difficulty-weighted mean of per-task scores:

    sc_alpha = Σ weight(tier(t))·score(t) / Σ weight(tier(t))

with tier weights {1 (hard): 3.0, 2 (medium): 1.0, 3 (easy): 0.5}. Tasks that
declare no tier default to medium.

This module is deliberately generic: it holds the weight table and weighting
math only. It carries no mapping of held-out task names to tiers.
"""

from __future__ import annotations

from collections.abc import Iterable

WEIGHTS = {1: 3.0, 2: 1.0, 3: 0.5}
DEFAULT_TIER = 2


def weight(tier: int | None) -> float:
    """Difficulty weight for a tier; falls back to the medium weight."""
    return WEIGHTS.get(tier if tier is not None else DEFAULT_TIER, WEIGHTS[DEFAULT_TIER])


def sc_alpha(pairs: Iterable[tuple[int | None, float]]) -> float:
    """Difficulty-weighted mean of (tier, score) pairs. Returns 0.0 on empty."""
    total_weight = 0.0
    weighted_sum = 0.0
    for tier, score in pairs:
        w = weight(tier)
        total_weight += w
        weighted_sum += w * score
    return weighted_sum / total_weight if total_weight else 0.0
