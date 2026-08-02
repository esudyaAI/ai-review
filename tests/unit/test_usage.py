from reviewer.llm.usage import ModelUsage, UsageTracker, cost_for, report_lines

PRICING = {
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "gpt-5.5-2026-04-23": {"input": 5.00, "output": 30.00, "cached_input": 0.50},
}


def test_tracker_accumulates_per_model_across_multiple_records():
    tracker = UsageTracker()
    tracker.record("gpt-5.4-mini-2026-03-17", prompt_tokens=1000, completion_tokens=200)
    tracker.record("gpt-5.4-mini-2026-03-17", prompt_tokens=500, completion_tokens=100)
    tracker.record("gpt-5.5-2026-04-23", prompt_tokens=2000, completion_tokens=300, cached_tokens=1500)

    by_model = tracker.by_model()
    assert by_model["gpt-5.4-mini-2026-03-17"] == ModelUsage(prompt_tokens=1500, completion_tokens=300)
    assert by_model["gpt-5.5-2026-04-23"] == ModelUsage(
        prompt_tokens=2000, completion_tokens=300, cached_tokens=1500
    )


def test_cost_for_computes_input_output_and_cached_prices():
    usage = ModelUsage(prompt_tokens=2000, completion_tokens=300, cached_tokens=1500)
    # 500 non-cached input @ 5.00/M + 1500 cached @ 0.50/M + 300 output @ 30.00/M
    expected = (500 * 5.00 + 1500 * 0.50 + 300 * 30.00) / 1_000_000
    assert cost_for("gpt-5.5-2026-04-23", usage, PRICING) == expected


def test_cost_for_falls_back_to_input_price_when_no_cached_price_given():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=200, cached_tokens=400)
    # No "cached_input" entry for this model - cached tokens priced at the input rate.
    expected = (1000 * 0.75 + 200 * 4.50) / 1_000_000
    assert cost_for("gpt-5.4-mini-2026-03-17", usage, PRICING) == expected


def test_cost_for_returns_none_when_model_has_no_price_entry():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=200)
    assert cost_for("some-unpriced-model", usage, PRICING) is None


def test_cost_for_returns_none_when_price_entry_is_incomplete():
    usage = ModelUsage(prompt_tokens=1000, completion_tokens=200)
    assert cost_for("m", usage, {"m": {"input": 1.0}}) is None


def test_report_lines_includes_cost_for_priced_model():
    by_model = {"gpt-5.4-mini-2026-03-17": ModelUsage(prompt_tokens=1000, completion_tokens=200)}
    stdout_lines, stderr_warnings = report_lines(by_model, PRICING)
    assert len(stdout_lines) == 1
    assert "gpt-5.4-mini-2026-03-17" in stdout_lines[0]
    assert "1000" in stdout_lines[0]
    assert "200" in stdout_lines[0]
    assert "$" in stdout_lines[0]
    assert stderr_warnings == []


def test_report_lines_warns_and_omits_cost_for_unpriced_model():
    by_model = {"some-unpriced-model": ModelUsage(prompt_tokens=1000, completion_tokens=200)}
    stdout_lines, stderr_warnings = report_lines(by_model, PRICING)
    assert len(stdout_lines) == 1
    assert "some-unpriced-model" in stdout_lines[0]
    assert "1000" in stdout_lines[0]
    assert "$" not in stdout_lines[0]
    assert len(stderr_warnings) == 1
    assert "some-unpriced-model" in stderr_warnings[0]
