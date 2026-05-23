"""Test PipelineStep validation — imported from production code."""

from tasks import PipelineStep, FinancialTasks
import pytest


def test_pipeline_step_creation():
    step = PipelineStep(name="Test Agent", agent="mock_agent", task={"input": "test"})
    assert step.name == "Test Agent"
    assert step.agent == "mock_agent"
    assert step.task["input"] == "test"
    assert step._rag_query_fn is None
    assert step._fallback_params == ""


def test_pipeline_step_with_rag():
    def mock_query(q: str) -> str:
        return f"result for {q}"

    step = PipelineStep(
        name="Valuation Specialist",
        agent="mock_agent",
        task={"input": "analyze {valuation_placeholder}"},
        _rag_query_fn=mock_query,
        _fallback_params="fallback text",
    )
    assert step._rag_query_fn is not None
    assert step._rag_query_fn("test") == "result for test"
    assert step._fallback_params == "fallback text"


class TestFinancialTasksBinding:
    """Test create_pipeline() validation logic from tasks.py."""

    EXPECTED_ROLES = [
        "Financial Document Analyst",
        "Business Model Analyst",
        "Growth & Revenue Analyst",
        "Valuation Specialist",
        "Senior Investment Advisor",
    ]

    @staticmethod
    def _make_mock_agent(role: str):
        return type("MockAgent", (), {"role": role})()

    def test_all_5_agents_succeeds(self):
        ft = FinancialTasks()
        agents = [self._make_mock_agent(r) for r in self.EXPECTED_ROLES]
        pipeline = ft.create_pipeline(
            agents=agents,
            extracted_docs={"report.pdf": "content"},
            valuation_params="some params",
            company_name="TestCorp",
        )
        assert len(pipeline) == 5

    def test_raises_on_wrong_agent_count(self):
        ft = FinancialTasks()
        agents = [self._make_mock_agent("Financial Document Analyst")]
        with pytest.raises(ValueError, match="Expected exactly 5 agents"):
            ft.create_pipeline(
                agents=agents,
                extracted_docs={},
                valuation_params="",
                company_name="X",
            )

    def test_raises_on_unrecognized_role(self):
        ft = FinancialTasks()
        agents = [self._make_mock_agent(r) for r in [
            "Financial Document Analyst",
            "Business Model Analyst",
            "Growth & Revenue Analyst",
            "Valuation Specialist",
            "Mystery Role",
        ]]
        with pytest.raises(ValueError, match="Unrecognized agent role"):
            ft.create_pipeline(
                agents=agents,
                extracted_docs={},
                valuation_params="",
                company_name="X",
            )

    def test_raises_on_duplicate_role(self):
        ft = FinancialTasks()
        agents = [self._make_mock_agent(r) for r in [
            "Financial Document Analyst",
            "Financial Document Analyst",
            "Growth & Revenue Analyst",
            "Valuation Specialist",
            "Senior Investment Advisor",
        ]]
        with pytest.raises(ValueError, match="Duplicate agent"):
            ft.create_pipeline(
                agents=agents,
                extracted_docs={},
                valuation_params="",
                company_name="X",
            )

    def test_raises_on_missing_role(self):
        ft = FinancialTasks()
        agents = [self._make_mock_agent(r) for r in [
            "Financial Document Analyst",
            "Business Model Analyst",
            "Growth & Revenue Analyst",
            "Valuation Specialist",
            "Financial Document Analyst",
        ]]
        with pytest.raises((ValueError,), match="Duplicate agent|Missing required"):
            ft.create_pipeline(
                agents=agents,
                extracted_docs={},
                valuation_params="",
                company_name="X",
            )
