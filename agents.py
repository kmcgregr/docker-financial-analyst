"""
Financial Analysis Agents
Defines all specialized AI agents for financial analysis using LangChain.

Fixes applied vs original:
  U1 - __init__ catches OllamaUnavailableError and ModelNotFoundError from
       check_model_availability() and re-raises as RuntimeError with context.
       Previously check_model_availability() called sys.exit(1) which killed
       the process silently with no traceback or actionable error message.

  U2 - Benefit inherited from utils.py: model names with digest suffixes
       (e.g. "gemma3:12b-it-qat-Q4_K_M") are now correctly recognised.

  U3 - Debug print statements removed from __init__. Model name and URL
       are logged cleanly without the "DEBUG:" prefix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable

from utils import check_model_availability, OllamaUnavailableError, ModelNotFoundError


@dataclass
class AnalystAgent:
    """Binds a role name to a Runnable chain (prompt | llm)."""
    role: str
    chain: Runnable

load_dotenv()


class FinancialAgents:
    """Factory class for creating financial analysis agents."""

    def __init__(self) -> None:
        """
        Initialize the agents factory with LLM configuration.

        U1: Catches typed exceptions from check_model_availability() and
            re-raises as RuntimeError so main.py's handler produces a full
            traceback with an actionable message instead of a silent exit.
        """
        analysis_model  = os.getenv("ANALYSIS_MODEL", "llama3.1:8b")
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # U3: clean startup log — no DEBUG: prefix
        print(f"  Analysis model:  {analysis_model}")
        print(f"  Ollama URL:      {ollama_base_url}")

        # U1: catch typed exceptions; re-raise with context for main.py
        try:
            check_model_availability(analysis_model)
        except OllamaUnavailableError as exc:
            raise RuntimeError(
                f"Cannot initialise Financial Agents — Ollama is not reachable.\n{exc}"
            ) from exc
        except ModelNotFoundError as exc:
            raise RuntimeError(
                f"Cannot initialise Financial Agents — analysis model missing.\n{exc}"
            ) from exc

        self.analysis_llm = Ollama(
            model=analysis_model,
            base_url=ollama_base_url,
            temperature=0.1,
            num_predict=4096,
            timeout=180,
        )

        # Available for optional injection by the orchestrator
        self.valuation_rag = None

    def set_valuation_rag(self, valuation_rag) -> None:
        """Set the valuation RAG instance for agents to use."""
        self.valuation_rag = valuation_rag

    # ------------------------------------------------------------------
    # Internal — agent builder
    # ------------------------------------------------------------------

    def _create_simple_agent(
        self, role: str, goal: str, backstory: str
    ) -> AnalystAgent:
        """Create an agent (prompt | llm runnable) wrapped with its role name."""

        template = (
            "You are an expert in financial analysis. Your role is {role}.\n"
            "Your main goal is: {goal}.\n\n"
            "Your backstory is:\n{backstory}\n\n"
            "IMPORTANT INSTRUCTIONS:\n"
            "- Analyze the provided financial data directly\n"
            "- Be thorough and detailed in your analysis\n"
            "- Use specific numbers and metrics from the data\n"
            "- Provide clear, actionable insights\n"
            "- Do NOT say you need more information — work with what you have\n"
            "- Format your response clearly with sections and bullet points "
            "where appropriate\n\n"
            "Question/Task:\n{input}\n\n"
            "Your detailed analysis:"
        )

        prompt = PromptTemplate(
            input_variables=["input"],
            template=template.format(
                role=role,
                goal=goal,
                backstory=backstory,
                input="{input}",
            ),
        )

        chain = prompt | self.analysis_llm
        return AnalystAgent(role=role, chain=chain)

    # ------------------------------------------------------------------
    # Individual agent constructors
    # ------------------------------------------------------------------

    def create_document_analyst(self) -> AnalystAgent:
        """Agent 1 — Financial Document Analyst."""
        return self._create_simple_agent(
            role="Financial Document Analyst",
            goal="Extract and organize comprehensive financial data from company reports",
            backstory=(
                "You are an expert financial document analyst with 15 years of "
                "experience reading and interpreting financial statements, quarterly "
                "reports, and annual reports. You have a meticulous eye for detail "
                "and can extract key financial metrics including revenue, expenses, "
                "profit margins, cash flows, and balance sheet items. You organize "
                "data systematically and ensure no critical information is missed. "
                "You're skilled at identifying trends across multiple reporting periods."
            ),
        )

    def create_business_analyst(self) -> AnalystAgent:
        """Agent 2 — Business Model Analyst."""
        return self._create_simple_agent(
            role="Business Model Analyst",
            goal=(
                "Analyze and clearly explain the company business model, "
                "revenue streams, and competitive positioning"
            ),
            backstory=(
                "You are a business strategy expert and former management consultant "
                "who has analyzed hundreds of companies across various industries. "
                "You excel at understanding how companies create and capture value, "
                "identifying their core competencies, and explaining complex business "
                "models in clear, accessible language. You can identify revenue "
                "streams, customer segments, competitive advantages (moats), and "
                "market positioning. You understand both B2B and B2C business models, "
                "subscription vs. transactional models, and various monetization "
                "strategies."
            ),
        )

    def create_growth_analyst(self) -> AnalystAgent:
        """Agent 3 — Growth & Revenue Analyst."""
        return self._create_simple_agent(
            role="Growth & Revenue Analyst",
            goal=(
                "Analyze revenue growth trends, KPIs, and pricing power "
                "to assess business momentum"
            ),
            backstory=(
                "You are a quantitative analyst specializing in growth metrics and "
                "revenue analysis. You have a strong background in statistics and "
                "financial modeling. You calculate and interpret growth rates (QoQ, "
                "YoY, CAGR), analyze key performance indicators specific to different "
                "business models, and assess pricing power through margin analysis. "
                "You can identify whether growth is organic or inorganic, sustainable "
                "or one-time, and evaluate the quality of revenue. You understand "
                "metrics like customer acquisition cost (CAC), lifetime value (LTV), "
                "retention rates, and other industry-specific KPIs."
            ),
        )

    def create_valuation_specialist(self) -> AnalystAgent:
        """Agent 4 — Valuation Specialist."""
        return self._create_simple_agent(
            role="Valuation Specialist",
            goal=(
                "Calculate comprehensive company valuation using multiple "
                "methodologies and provided parameters"
            ),
            backstory=(
                "You are a CFA charterholder and valuation expert with deep "
                "expertise in multiple valuation methodologies including Discounted "
                "Cash Flow (DCF), comparable company analysis (comps), precedent "
                "transactions, and various multiples-based approaches (P/E, EV/EBITDA, "
                "P/S, P/B, etc.). You understand how to apply appropriate methodologies "
                "based on company stage, industry, and available data. You can "
                "interpret and apply valuation parameters, adjust for risk factors, "
                "and synthesize multiple approaches into a fair value range. You're "
                "skilled at explaining the rationale behind valuation assumptions and "
                "highlighting key value drivers."
            ),
        )

    def create_investment_advisor(self) -> AnalystAgent:
        """Agent 5 — Senior Investment Advisor."""
        return self._create_simple_agent(
            role="Senior Investment Advisor",
            goal="Synthesize all analyses into a clear, actionable investment recommendation",
            backstory=(
                "You are a senior investment advisor with 20+ years of experience in "
                "equity research and portfolio management. You have successfully guided "
                "investors through multiple market cycles and have a track record of "
                "identifying both great opportunities and potential pitfalls. You "
                "synthesize fundamental analysis, business quality assessment, growth "
                "prospects, and valuation to form holistic investment opinions. You "
                "provide clear BUY, HOLD, or SELL recommendations with well-reasoned "
                "supporting arguments. You balance optimism with skepticism, always "
                "considering both the bull and bear cases. You communicate in a direct, "
                "actionable manner while being honest about uncertainties and risks."
            ),
        )

    # ------------------------------------------------------------------
    # Factory — returns all 5 agents in pipeline order
    # ------------------------------------------------------------------

    def create_agents(self) -> List[AnalystAgent]:
        """
        Create and return all financial analysis agents in execution order.

        Returns exactly 5 agents. create_pipeline() in tasks.py validates
        this count and raises immediately if it differs.
        """
        print("  Creating analysis agents...")

        agents = [
            self.create_document_analyst(),
            self.create_business_analyst(),
            self.create_growth_analyst(),
            self.create_valuation_specialist(),
            self.create_investment_advisor(),
        ]

        print(f"  ✓ {len(agents)} agents created")
        return agents