from __future__ import annotations

import json
import math
from collections.abc import Iterator

from heist.models import CostProvenance, CostSource, TaskRunResult, TokenUsage, UsageCapture

INPUT_KEYS = {"input_tokens", "inputTokens", "prompt_tokens", "tokens_in", "input"}
OUTPUT_KEYS = {"output_tokens", "outputTokens", "completion_tokens", "tokens_out", "output"}
CACHE_READ_KEYS = {
    "cache_read_tokens",
    "cacheReadTokens",
    "cache_read_input_tokens",
    "cacheReadInputTokens",
    "cached_input_tokens",
    "cachedInputTokens",
    "cache_read",
    "cacheRead",
}
# OpenAI-style: input_tokens is the TOTAL prompt size and cached_input_tokens
# is a subset of it. To keep TokenUsage.input meaning "non-cached input" across
# providers, subtract these keys from the input count at capture time.
OPENAI_SUBSET_CACHE_KEYS = {"cached_input_tokens", "cachedInputTokens"}
CACHE_WRITE_KEYS = {
    "cache_write_tokens",
    "cacheWriteTokens",
    "cache_creation_input_tokens",
    "cacheCreationInputTokens",
    "cache_write",
    "cacheWrite",
}
REPORTED_TOTAL_COST_KEYS = {"total_cost_usd", "totalCostUsd", "total_cost", "totalCost"}
REPORTED_COST_KEYS = {"cost_usd", "costUSD", "cost"}


def _objects(value: object) -> Iterator[dict[str, object]]:
    # Iterative walk so a pathologically nested JSON payload from an agent
    # can't blow the recursion stack (default ~1000) and crash the worker
    # — which would propagate up and take the whole parallel pool down.
    stack: list[object] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            yield current
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)


def _int_value(value: object) -> int:
    # Accept int, float, and stringified-int variants. Some gateways pass token
    # counts through as strings (especially when proxying SSE-to-JSON); silently
    # zeroing those out would corrupt cost reconstruction. Negative values are
    # clamped to 0 — defensive against buggy providers reporting credits.
    # Token counts are integral in practice; round() (round-half-to-even) only
    # matters for the fractional inputs that providers don't actually emit.
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        if value < 0:
            return 0
        return round(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return 0
        if parsed < 0:
            return 0
        return round(parsed)
    return 0


def _float_value(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        parsed = float(value)
        if math.isfinite(parsed) and parsed >= 0:
            return parsed
        return None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if math.isfinite(parsed) and parsed >= 0:
            return parsed
    return None


def _model_usage_costs(value: object) -> Iterator[tuple[float, str]]:
    if not isinstance(value, dict):
        return
    for model_id, model_usage in value.items():
        if not isinstance(model_usage, dict):
            continue
        cost = _float_value(model_usage.get("costUSD"))
        if cost is not None:
            yield cost, f"modelUsage.{model_id}.costUSD"


def _model_usage_total(value: object) -> tuple[float, str] | None:
    if not isinstance(value, dict):
        return None

    total = 0.0
    sources: list[str] = []
    for cost, source in _model_usage_costs(value):
        total += cost
        sources.append(source)
    if not sources:
        return None
    return total, "+".join(sources)


# Cost priority tiers, lowest = most authoritative.
_TIER_TOTAL = 0  # explicit session total (total_cost_usd) — cumulative
_TIER_MODEL_USAGE = 1  # sum of per-model costUSD — per-turn
_TIER_BARE_COST = 2  # a bare cost_usd / cost key — per-turn


def _reported_cost_tier(key: object) -> int | None:
    if key in REPORTED_TOTAL_COST_KEYS:
        return _TIER_TOTAL
    if key in REPORTED_COST_KEYS:
        return _TIER_BARE_COST
    return None


def _line_cost(payload: object) -> dict[int, tuple[float, str]]:
    """Best reported cost per priority tier within a single JSON line.

    Within one line the same value often appears at several nesting levels, so
    repeats are deduped by taking the max per tier. The ``modelUsage`` subtree is
    aggregated once at tier 1 and NOT descended into, so its inner ``costUSD``
    values aren't also counted as tier-2 bare costs (which would double-count).
    Iterative walk so a pathologically nested payload can't blow the stack.
    """
    tiers: dict[int, tuple[float, str]] = {}

    def offer(tier: int, cost: float, source: str) -> None:
        current = tiers.get(tier)
        if current is None or cost > current[0]:
            tiers[tier] = (cost, source)

    stack: list[object] = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            model_usage_total = _model_usage_total(current.get("modelUsage"))
            if model_usage_total is not None:
                offer(_TIER_MODEL_USAGE, *model_usage_total)
            for key, child in current.items():
                if key == "modelUsage":
                    continue  # aggregated above; don't descend (avoids double-count)
                cost = _float_value(child)
                tier = _reported_cost_tier(key)
                if cost is not None and tier is not None:
                    offer(tier, cost, str(key))
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
    return tiers


def _max_int(obj: dict[str, object], keys: set[str]) -> int:
    return max(0, *(_int_value(obj.get(key)) for key in keys))


def _per_line_usage(payload: object) -> tuple[int, int, int, int]:
    """Collapse all nested usage objects in a single JSON line down to one
    `(input, output, cache_read, cache_write)` tuple. Within a single line,
    the same usage dict often appears at multiple nesting levels (e.g. once
    under `event.usage` and again under `modelUsage[model].usage`); taking the
    max across nested objects deduplicates without losing values.

    `capture_usage` then sums the per-line tuples across the whole stream. This
    handles all three observed provider shapes correctly:
      - codex/cursor: one final usage line → sum == max == correct total
      - claude (Anthropic stream-json): one usage event per turn, multiple
        turns per task → summing across lines gives the true total
    """
    best_input = 0
    best_output = 0
    best_cache_read = 0
    best_cache_write = 0
    for obj in _objects(payload):
        obj_input = _max_int(obj, INPUT_KEYS)
        obj_output = _max_int(obj, OUTPUT_KEYS)
        obj_cache_read = _max_int(obj, CACHE_READ_KEYS)
        obj_cache_write = _max_int(obj, CACHE_WRITE_KEYS)
        subset_cache = _max_int(obj, OPENAI_SUBSET_CACHE_KEYS)
        if subset_cache > 0 and obj_input >= subset_cache:
            obj_input -= subset_cache
        best_input = max(best_input, obj_input)
        best_output = max(best_output, obj_output)
        best_cache_read = max(best_cache_read, obj_cache_read)
        best_cache_write = max(best_cache_write, obj_cache_write)
    return best_input, best_output, best_cache_read, best_cache_write


def capture_usage(text: str) -> UsageCapture:
    usage = TokenUsage()
    # Cost aggregation mirrors token aggregation (summed across lines) so a
    # multi-turn session isn't under-counted. Tier 0 (explicit session total) is
    # cumulative, so take its max across lines; tiers 1/2 (per-turn modelUsage /
    # bare cost) are summed across lines. Tier 0 wins when present.
    total_cost: float | None = None
    total_source: str | None = None
    per_turn_sum = 0.0
    per_turn_tier: int | None = None
    per_turn_source: str | None = None
    for line in text.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        line_input, line_output, line_cache_read, line_cache_write = _per_line_usage(payload)
        usage.input += line_input
        usage.output += line_output
        usage.cache_read += line_cache_read
        usage.cache_write += line_cache_write

        tiers = _line_cost(payload)
        if _TIER_TOTAL in tiers:
            cost, source = tiers[_TIER_TOTAL]
            if total_cost is None or cost > total_cost:
                total_cost = cost
                total_source = source
        # Within a line prefer the modelUsage aggregate over a bare cost key —
        # they describe the same spend two ways; counting both double-counts.
        turn_tiers = [t for t in (_TIER_MODEL_USAGE, _TIER_BARE_COST) if t in tiers]
        if turn_tiers:
            tier = min(turn_tiers)
            cost, source = tiers[tier]
            per_turn_sum += cost
            if per_turn_tier is None or tier < per_turn_tier:
                per_turn_tier = tier
                per_turn_source = source

    if total_cost is not None:
        return UsageCapture(
            usage=usage, reported_cost_usd=total_cost, reported_cost_source=total_source
        )
    if per_turn_tier is not None:
        return UsageCapture(
            usage=usage, reported_cost_usd=per_turn_sum, reported_cost_source=per_turn_source
        )
    return UsageCapture(usage=usage)


# (input, output, cache_read, cache_write) in USD per million tokens.
# `inputTokens` and `cacheReadTokens` are treated as disjoint pools, matching
# cursor-agent's Anthropic-style stream-json field naming.
#
# PRICING_LAST_VERIFIED: 2026-06-14
# Sources:
#   gpt-5.5            — https://openai.com/api/pricing
#   gpt-5.4-mini       — https://developers.openai.com/api/docs/models/gpt-5.4-mini
#   claude-*           — https://www.anthropic.com/pricing
#   composer-*         — https://cursor.com/docs/models-and-pricing
#   composer-2.5       — https://cursor.com/changelog/composer-2-5
#   grok-4.3           — https://docs.x.ai/docs/models
#   kimi-k2.5          — https://platform.moonshot.ai/docs/pricing
#   gemini-3.5-flash   — https://ai.google.dev/gemini-api/docs/pricing
#   openrouter/*       — https://openrouter.ai (opencode model catalog, 2026-06-14)
PRICING_PER_MILLION: dict[str, tuple[float, float, float, float]] = {
    "gpt-5.5": (5.0, 30.0, 0.5, 0.0),
    "gpt-5.4-mini": (0.75, 4.50, 0.075, 0.0),
    "claude-opus-4-8": (5.0, 25.0, 0.5, 6.25),
    "claude-opus-4-7": (5.0, 25.0, 0.5, 6.25),
    "claude-sonnet-4-6": (3.0, 15.0, 0.3, 3.75),
    "claude-haiku-4-5": (1.0, 5.0, 0.1, 1.25),
    "composer-2.5": (0.50, 2.50, 0.20, 0.0),
    "grok-4.3": (1.25, 2.50, 0.20, 0.0),
    "kimi-k2.5": (0.60, 3.00, 0.10, 0.0),
    "gemini-3.5-flash": (1.50, 9.00, 0.15, 0.0),
    "openrouter/google/gemini-3.5-flash": (1.50, 9.00, 0.20, 0.0),
    "openrouter/deepseek/deepseek-v4-pro": (0.87, 3.48, 0.20, 0.0),
    "openrouter/moonshotai/kimi-k2.6": (0.68, 3.41, 0.34, 0.0),
    "openrouter/qwen/qwen-2.5-coder-32b-instruct": (0.66, 1.0, 0.0, 0.0),
    "openrouter/qwen/qwen3.7-max": (1.25, 3.75, 0.0, 0.0),
}


def cost_for_usage(model_id: str, usage: TokenUsage) -> float | None:
    pricing = PRICING_PER_MILLION.get(model_id)
    if not pricing:
        return None
    if usage.input == usage.output == usage.cache_read == usage.cache_write == 0:
        return None
    input_price, output_price, cache_read_price, cache_write_price = pricing
    million = 1_000_000
    return (
        usage.input * input_price
        + usage.output * output_price
        + usage.cache_read * cache_read_price
        + usage.cache_write * cache_write_price
    ) / million


def choose_cost(
    model_id: str, capture: UsageCapture
) -> tuple[float | None, CostSource, float | None, CostProvenance]:
    reconstructed = cost_for_usage(model_id, capture.usage)
    if capture.reported_cost_usd is not None:
        provenance: CostProvenance = (
            "reconciled" if reconstructed is not None else "as_reported_only"
        )
        return capture.reported_cost_usd, "reported", reconstructed, provenance
    if reconstructed is not None:
        return reconstructed, "reconstructed", reconstructed, "reconciled"
    return None, "unavailable", None, "cost_not_available"


def primary_cost(result: TaskRunResult) -> float | None:
    """Best per-task cost: explicit cost first, then reconstructed, then reported total."""
    if result.cost_usd is not None:
        return result.cost_usd
    if result.reconstructed_per_task_cost_usd is not None:
        return result.reconstructed_per_task_cost_usd
    return result.reported_session_cost_usd


def cost_source_label(result: TaskRunResult) -> str:
    """Human label for which cost field primary_cost picked."""
    if result.cost_source != "unavailable":
        return result.cost_source
    if result.reconstructed_per_task_cost_usd is not None:
        return "reconstructed"
    if result.reported_session_cost_usd is not None:
        return "reported"
    return "unavailable"
