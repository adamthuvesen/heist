from __future__ import annotations

from heist.agents import DEFAULT_AGENTS
from heist.usage import PRICING_PER_MILLION


def test_every_default_agent_has_a_cost_source() -> None:
    """Every shipped agent must have a pricing entry in PRICING_PER_MILLION.

    If this fails, add the model to PRICING_PER_MILLION in src/heist/usage.py
    with the verified per-million-token rates and a source URL.
    """
    missing = [
        f"{agent_id} (model_id={spec.model_id})"
        for agent_id, spec in DEFAULT_AGENTS.items()
        if spec.model_id not in PRICING_PER_MILLION
    ]

    assert not missing, (
        "Agents with no cost source: "
        + ", ".join(sorted(missing))
        + ". Add to PRICING_PER_MILLION."
    )
