from __future__ import annotations

import pytest

from heist.models import TokenUsage, UsageCapture
from heist.usage import PRICING_PER_MILLION, capture_usage, choose_cost, cost_for_usage


def test_capture_usage_reads_codex_jsonl_shape() -> None:
    # OpenAI reports input_tokens as TOTAL (cached + non-cached); the captured
    # input value should be the non-cached portion only so cost reconstruction
    # does not double-charge the cache hits.
    capture = capture_usage(
        """
{"type":"turn.completed","usage":{"input_tokens":715476,"cached_input_tokens":400000,"output_tokens":13068}}
"""
    )

    assert capture.usage.input == 315476
    assert capture.usage.output == 13068
    assert capture.usage.cache_read == 400000
    assert capture.reported_cost_usd is None


def test_cost_for_usage_does_not_double_charge_openai_cached_input() -> None:
    capture = capture_usage(
        """
{"type":"turn.completed","usage":{"input_tokens":1350631,"cached_input_tokens":1257984,"output_tokens":14959}}
"""
    )

    cost = cost_for_usage("gpt-5.5", capture.usage)

    # 92,647 non-cached input @ $5/M + 1,257,984 cache reads @ $0.5/M + 14,959 output @ $30/M.
    assert cost is not None
    assert round(cost, 4) == round((92_647 * 5 + 1_257_984 * 0.5 + 14_959 * 30) / 1_000_000, 4)


def test_cost_for_usage_prices_disjoint_anthropic_cache_separately() -> None:
    # Anthropic/cursor stream-json reports `input` and `cache_read` as DISJOINT
    # pools. Cost must price cache_read at its own rate, never folding it into
    # input (which would double-charge). This locks the invariant documented
    # above PRICING_PER_MILLION so a future "fix" that subtracts cache_read from
    # input for these providers is caught.
    model = "claude-opus-4-8"
    in_p, out_p, cr_p, cw_p = PRICING_PER_MILLION[model]
    usage = TokenUsage(
        input=1_000_000,
        output=200_000,
        cache_read=3_000_000,
        cache_write=50_000,
    )
    cost = cost_for_usage(model, usage)
    expected = (1_000_000 * in_p + 200_000 * out_p + 3_000_000 * cr_p + 50_000 * cw_p) / 1_000_000
    assert cost == pytest.approx(expected)
    # And cache_read genuinely contributes its own line item (not zero, not
    # absorbed into input): dropping it lowers the cost by exactly cr_p per token.
    assert cost_for_usage(model, usage.model_copy(update={"cache_read": 0})) == pytest.approx(
        expected - 3_000_000 * cr_p / 1_000_000
    )


def test_capture_usage_reads_claude_cost_and_cache_tokens() -> None:
    capture = capture_usage(
        """
{"type":"result","total_cost_usd":2.023388,"usage":{"input_tokens":22,"output_tokens":44742,"cache_creation_input_tokens":69642,"cache_read_input_tokens":938931},"modelUsage":{"claude-opus-4-7":{"costUSD":2.023388}}}
"""
    )

    assert capture.usage.input == 22
    assert capture.usage.output == 44742
    assert capture.usage.cache_write == 69642
    assert capture.usage.cache_read == 938931
    assert capture.reported_cost_usd == 2.023388
    assert capture.reported_cost_source == "total_cost_usd"


def test_capture_usage_reads_cursor_camel_case_tokens() -> None:
    capture = capture_usage(
        """
{"type":"result","usage":{"inputTokens":42512,"outputTokens":18369,"cacheReadTokens":569344,"cacheWriteTokens":12}}
"""
    )

    assert capture.usage.input == 42512
    assert capture.usage.output == 18369
    assert capture.usage.cache_read == 569344
    assert capture.usage.cache_write == 12


def test_capture_usage_can_sum_model_usage_costs_without_total() -> None:
    capture = capture_usage(
        """
{"type":"result","modelUsage":{"model-a":{"costUSD":1.25},"model-b":{"costUSD":0.75}}}
"""
    )

    assert capture.reported_cost_usd == 2.0
    assert capture.reported_cost_source == "modelUsage.model-a.costUSD+modelUsage.model-b.costUSD"


def test_capture_usage_accepts_stringified_reported_costs() -> None:
    capture = capture_usage(
        """
{"type":"result","total_cost_usd":"1.75","modelUsage":{"model-a":{"costUSD":"0.50"}}}
"""
    )

    assert capture.reported_cost_usd == 1.75
    assert capture.reported_cost_source == "total_cost_usd"


def test_capture_usage_rejects_non_finite_reported_costs() -> None:
    capture = capture_usage(
        """
{"type":"result","total_cost_usd":"inf","cost_usd":"nan"}
"""
    )

    assert capture.reported_cost_usd is None
    assert capture.reported_cost_source is None


def test_choose_cost_prefers_reported_cost() -> None:
    capture = UsageCapture(
        usage=TokenUsage(input=100, output=20),
        reported_cost_usd=1.23,
        reported_cost_source="total_cost_usd",
    )

    cost, source, reconstructed, provenance = choose_cost("gpt-5.5", capture)

    assert cost == 1.23
    assert source == "reported"
    assert reconstructed == cost_for_usage("gpt-5.5", capture.usage)
    assert provenance == "reconciled"


def test_choose_cost_falls_back_to_reconstructed_cost() -> None:
    capture = UsageCapture(usage=TokenUsage(input=100, output=20))

    cost, source, reconstructed, provenance = choose_cost("gpt-5.5", capture)

    assert cost == 0.0011
    assert source == "reconstructed"
    assert reconstructed == 0.0011
    assert provenance == "reconciled"


def test_composer_25_standard_pricing_matches_verified_cursor_rates() -> None:
    usage = TokenUsage(input=1_000_000, output=1_000_000, cache_read=1_000_000)

    cost = cost_for_usage("composer-2.5", usage)

    assert cost == 3.20


def test_capture_usage_reads_opencode_reported_cost() -> None:
    capture = capture_usage(
        """
{"type":"result","total_cost_usd":0.42,"usage":{"inputTokens":1200,"outputTokens":800}}
"""
    )

    assert capture.usage.input == 1200
    assert capture.usage.output == 800
    assert capture.reported_cost_usd == 0.42
    assert capture.reported_cost_source == "total_cost_usd"


def test_choose_cost_marks_unknown_pricing_unavailable() -> None:
    capture = UsageCapture(usage=TokenUsage(input=100, output=20))

    cost, source, reconstructed, provenance = choose_cost("unknown-model", capture)

    assert cost is None
    assert source == "unavailable"
    assert reconstructed is None
    assert provenance == "cost_not_available"


def test_capture_usage_skips_non_json_lines_without_aborting() -> None:
    # Real agent stdout interleaves human-readable status with usage JSON. A
    # JSON decode error mid-stream must not lose the usage event further down.
    capture = capture_usage(
        """
Initialising codex...
{"not": "usage"}
{"type":"turn.completed","usage":{"input_tokens":100,"output_tokens":20}}
"""
    )

    assert capture.usage.input == 100
    assert capture.usage.output == 20


def test_capture_usage_treats_negative_counts_as_zero() -> None:
    # Defensive: a buggy provider emitting negative token counts shouldn't
    # propagate into the report as a credit.
    capture = capture_usage('{"usage":{"input_tokens":-50,"output_tokens":-10}}')
    assert capture.usage.input == 0
    assert capture.usage.output == 0


def test_capture_usage_treats_bool_token_counts_as_zero() -> None:
    capture = capture_usage('{"usage":{"input_tokens":true,"output_tokens":false}}')
    # bool(True) is technically an int(1) in Python; the parser explicitly
    # rejects bool to avoid spurious 1-token attributions.
    assert capture.usage.input == 0
    assert capture.usage.output == 0


def test_capture_usage_handles_cached_greater_than_input_without_underflow() -> None:
    # Degenerate but real-world: a gateway may report cached_input_tokens > input
    # if the cache field is the total tokens and input is the new-only delta.
    # The OpenAI subset-cache guard inside `_per_line_usage` means we don't
    # subtract in that case — we keep input as-is and record cache_read alongside.
    capture = capture_usage(
        '{"usage":{"input_tokens":50,"cached_input_tokens":200,"output_tokens":10}}'
    )
    assert capture.usage.input == 50
    assert capture.usage.cache_read == 200


def test_capture_usage_sums_per_turn_deltas_across_events() -> None:
    # C1 invariant: Anthropic-style stream-json emits one usage event per turn
    # for a multi-turn task. The captured totals must SUM across events, not
    # silently pick the max (which would under-count by N turns - 1).
    stream = "\n".join(
        '{"type":"message_delta","usage":{"input_tokens":3,"output_tokens":250,'
        '"cache_creation_input_tokens":1000,"cache_read_input_tokens":500}}'
        for _ in range(4)
    )

    capture = capture_usage(stream)

    # 4 turns × (3 input, 250 output, 1000 cache_write, 500 cache_read).
    assert capture.usage.input == 12
    assert capture.usage.output == 1000
    assert capture.usage.cache_write == 4000
    assert capture.usage.cache_read == 2000


def test_capture_usage_single_final_summary_is_not_doubled() -> None:
    # Codex/cursor-style: a single line with one usage summary. The new
    # sum-across-lines logic must still treat that as N=1 (no double-counting
    # from nested duplicates within the same line).
    stream = (
        '{"type":"turn.completed","usage":{"input_tokens":1000,'
        '"output_tokens":200},"modelUsage":{"gpt-5.5":'
        '{"usage":{"input_tokens":1000,"output_tokens":200}}}}'
    )

    capture = capture_usage(stream)

    # Same usage object is wrapped twice within the line. Sum across lines = 1
    # line, so total == single observed value (max-per-line dedupes).
    assert capture.usage.input == 1000
    assert capture.usage.output == 200
