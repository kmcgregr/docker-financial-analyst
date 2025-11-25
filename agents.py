"""
Financial Analysis Agents
Defines all specialized AI agents for financial analysis using LangChain
"""
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from typing import List, Dict, Any
import os
from utils import check_model_availability
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class FinancialAgents:
    """Factory class for creating financial analysis agents"""

    def __init__(self):
        """Initialize the agents factory with LLM configuration"""

        # Get model configuration from environment
        analysis_model = os.getenv('ANALYSIS_MODEL', 'llama3.1:8b')
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')

        # Debug: Print what we got from environment
        print(f"  DEBUG: ANALYSIS_MODEL from env = '{os.getenv('ANALYSIS_MODEL')}'")
        print(f"  DEBUG: Using analysis model: {analysis_model}")

        # Check if the model is available
        check_model_availability(analysis_model)

        # Initialize analysis LLM using Ollama
        self.analysis_llm = Ollama(
            model=analysis_model,
            base_url=ollama_base_url,
            temperature=0.1,  # Low temperature for more consistent financial analysis
        )

        print(f"  Using analysis model: {analysis_model}")
        print(f"  Ollama URL: {ollama_base_url}")
        
        # Store valuation RAG for tool access
        self.valuation_rag = None

    def set_valuation_rag(self, valuation_rag):
        """Set the valuation RAG instance for agents to use"""
        self.valuation_rag = valuation_rag

    def _create_simple_agent(self, role: str, goal: str, backstory: str) -> LLMChain:
        """Helper function to create a simple LLM chain agent"""

        # Create a simple prompt template without complex ReAct logic
        template = """You are an expert in financial analysis. Your role is {role}.
Your main goal is: {goal}.

Your backstory is:
{backstory}

IMPORTANT INSTRUCTIONS:
- Analyze the provided financial data directly
- Be thorough and detailed in your analysis
- Use specific numbers and metrics from the data
- Provide clear, actionable insights
- Do NOT say you need more information - work with what you have
- Format your response clearly with sections and bullet points where appropriate

Question/Task:
{input}

Your detailed analysis:"""

        prompt = PromptTemplate(
            input_variables=["input"],
            template=template.format(role=role, goal=goal, backstory=backstory, input="{input}")
        )

        # Create a simple LLM chain
        chain = LLMChain(
            llm=self.analysis_llm,
            prompt=prompt,
            verbose=True
        )

        return chain

    def create_document_analyst(self) -> LLMChain:
        """Create the Document Analyst agent"""
        return self._create_simple_agent(
            role='Financial Document Analyst',
            goal='Extract and organize comprehensive financial data from company reports',
            backstory="""You are an expert financial document analyst with 15 years of
            experience reading and interpreting financial statements, quarterly reports,
            and annual reports. You have a meticulous eye for detail and can extract
            key financial metrics including revenue, expenses, profit margins, cash flows,
            and balance sheet items. You organize data systematically and ensure no
            critical information is missed. You're skilled at identifying trends across
            multiple reporting periods."""
        )

    def create_business_analyst(self) -> LLMChain:
        """Create the Business Model Analyst agent"""
        return self._create_simple_agent(
            role='Business Model Analyst',
            goal='Analyze and clearly explain the company business model, revenue streams, and competitive positioning',
            backstory="""You are a business strategy expert and former management
            consultant who has analyzed hundreds of companies across various industries.
            You excel at understanding how companies create and capture value, identifying
            their core competencies, and explaining complex business models in clear,
            accessible language. You can identify revenue streams, customer segments,
            competitive advantages (moats), and market positioning. You understand both
            B2B and B2C business models, subscription vs. transactional models, and
            various monetization strategies."""
        )

    def create_growth_analyst(self) -> LLMChain:
        """Create the Growth & Revenue Analyst agent"""
        return self._create_simple_agent(
            role='Growth & Revenue Analyst',
            goal='Analyze revenue growth trends, KPIs, and pricing power to assess business momentum',
            backstory="""You are a quantitative analyst specializing in growth metrics
            and revenue analysis. You have a strong background in statistics and financial
            modeling. You calculate and interpret growth rates (QoQ, YoY, CAGR), analyze
            key performance indicators specific to different business models, and assess
            pricing power through margin analysis. You can identify whether growth is
            organic or inorganic, sustainable or one-time, and evaluate the quality of
            revenue. You understand metrics like customer acquisition cost (CAC),
            lifetime value (LTV), retention rates, and other industry-specific KPIs."""
        )

    def create_valuation_specialist(self) -> LLMChain:
        """Create the Valuation Specialist agent"""
        return self._create_simple_agent(
            role='Valuation Specialist',
            goal='Calculate comprehensive company valuation using multiple methodologies and provided parameters',
            backstory="""You are a CFA charterholder and valuation expert with deep
            expertise in multiple valuation methodologies including Discounted Cash Flow
            (DCF), comparable company analysis (comps), precedent transactions, and
            various multiples-based approaches (P/E, EV/EBITDA, P/S, P/B, etc.). You
            understand how to apply appropriate methodologies based on company stage,
            industry, and available data. You can interpret and apply valuation parameters,
            adjust for risk factors, and synthesize multiple approaches into a fair value
            range. You're skilled at explaining the rationale behind valuation assumptions
            and highlighting key value drivers."""
        )

    def create_investment_advisor(self) -> LLMChain:
        """Create the Senior Investment Advisor agent"""
        return self._create_simple_agent(
            role='Senior Investment Advisor',
            goal='Synthesize all analyses into a clear, actionable investment recommendation',
            backstory="""You are a senior investment advisor with 20+ years of experience
            in equity research and portfolio management. You have successfully guided
            investors through multiple market cycles and have a track record of identifying
            both great opportunities and potential pitfalls. You synthesize fundamental
            analysis, business quality assessment, growth prospects, and valuation to form
            holistic investment opinions. You provide clear BUY, HOLD, or SELL recommendations
            with well-reasoned supporting arguments. You balance optimism with skepticism,
            always considering both the bull and bear cases. You communicate in a direct,
            actionable manner while being honest about uncertainties and risks. You understand
            that the best investment decisions consider not just current metrics but also
            future potential and downside protection."""
        )

    def create_agents(self) -> List[LLMChain]:
        """Create and return all financial analysis agents in execution order"""
        
        print(f"  Creating simplified agents (no complex tool usage)")
        
        agents = [
            self.create_document_analyst(),
            self.create_business_analyst(),
            self.create_growth_analyst(),
            self.create_valuation_specialist(),
            self.create_investment_advisor()
        ]
        
        return agents