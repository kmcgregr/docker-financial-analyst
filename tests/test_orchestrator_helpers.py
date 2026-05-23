"""Test orchestrator helper functions — imported from production code."""

from utils import (
    extract_summary_block,
    cap_context,
    build_valuation_rag_query,
    FALLBACK_VALUATION_QUERY,
    MAX_CONTEXT_CHARS,
)


class TestExtractSummaryBlock:
    def test_extracts_summary_block(self):
        text = """some text
=== SUMMARY ===
KEY_METRICS: Revenue: $234M, +23% YoY
KEY_FINDINGS: Strong growth
CONFIDENCE: HIGH
=== END SUMMARY ===
more text"""
        result = extract_summary_block(text)
        assert "KEY_METRICS:" in result
        assert "KEY_FINDINGS:" in result
        assert "CONFIDENCE: HIGH" in result

    def test_falls_back_to_last_paragraph(self):
        text = """Paragraph one.

Paragraph two with key info.

Final paragraph here."""
        result = extract_summary_block(text)
        assert result == "Final paragraph here."

    def test_falls_back_to_suffix_when_no_paragraphs(self):
        text = "Single line no paragraphs."
        result = extract_summary_block(text)
        assert result == text[-500:]

    def test_empty_string(self):
        result = extract_summary_block("")
        assert result == ""


class TestCapContext:
    def test_does_not_truncate_short_text(self):
        short = "a" * 500
        assert cap_context(short) == short

    def test_truncates_long_text(self):
        long = "a\n" + "b" * MAX_CONTEXT_CHARS + "c"
        result = cap_context(long)
        assert len(result) <= MAX_CONTEXT_CHARS + 100
        assert "truncated" in result

    def test_preserves_newline_boundary(self):
        text = "head\n" + "m" * (MAX_CONTEXT_CHARS - 10) + "\n" + "tail"
        result = cap_context(text)
        assert "tail" in result


class TestBuildValuationRagQuery:
    def test_builds_query_from_summary(self):
        context = """=== SUMMARY ===
KEY_METRICS: Revenue: $100M, Gross Margin: 70%
KEY_FINDINGS: SaaS model, high retention
CONFIDENCE: HIGH
=== END SUMMARY ==="""
        query = build_valuation_rag_query(context)
        assert "What valuation methodologies" in query
        assert "SaaS model" in query

    def test_falls_back_when_no_summary(self):
        context = "Some random text without summary markers"
        query = build_valuation_rag_query(context)
        assert query == FALLBACK_VALUATION_QUERY

    def test_falls_back_when_no_metrics_or_findings(self):
        context = """=== SUMMARY ===
CONFIDENCE: MEDIUM
=== END SUMMARY ==="""
        query = build_valuation_rag_query(context)
        assert query == FALLBACK_VALUATION_QUERY
