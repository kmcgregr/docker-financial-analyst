"""
Financial Analysis Agents
Defines all specialized AI agents for financial analysis
"""

from crewai import Agent
from langchain_community.llms import Ollama
from typing import List
import os


class FinancialAgents:
    """Factory class for creating financial analysis agents"""
    
    def __init__(self):
        """Initialize the agents factory with LLM configuration"""
        
        # Set a dummy OpenAI API key to avoid CrewAI errors
        os.environ["OPENAI_API_KEY"] = "NA"
        
        # Get model configuration from environment
        analysis_model = os.getenv('ANALYSIS_MODEL', 'llama3.1:8b')
        ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
        
        # Initialize analysis LLM
        self.analysis_llm = Ollama(
            model=analysis_model,
            base_url=ollama_base_url,
            temperature=0.1  # Low temperature for more consistent financial analysis
        )
        
        print(f"  Using analysis model: {analysis_model}")
        print(f"  Ollama URL: {ollama_base_url}")
    
    def create_document_analyst(self) -> Agent:
        """Create the Document Analyst agent"""
        return Agent(
            role='Financial Document Analyst',
            goal='Extract and organize comprehensive financial data from company reports',
            backstory="""You are an expert financial document analyst with 15 years of 
            experience reading and interpreting financial statements, quarterly reports, 
            and annual reports. You have a meticulous eye for detail and can extract 
            key financial metrics including revenue, expenses, profit margins, cash flows, 
            and balance sheet items. You organize data systematically and ensure no 
            critical information is missed. You're skilled at identifying trends across 
            multiple reporting periods.""",
            llm=self.analysis_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )
    
    def create_business_analyst(self) -> Agent:
        """Create the Business Model Analyst agent"""
        return Agent(
            role='Business Model Analyst',
            goal='Analyze and clearly explain the company business model, revenue streams, and competitive positioning',
            backstory="""You are a business strategy expert and former management 
            consultant who has analyzed hundreds of companies across various industries. 
            You excel at understanding how companies create and capture value, identifying 
            their core competencies, and explaining complex business models in clear, 
            accessible language. You can identify revenue streams, customer segments, 
            competitive advantages (moats), and market positioning. You understand both 
            B2B and B2C business models, subscription vs. transactional models, and 
            various monetization strategies.""",
            llm=self.analysis_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )
    
    def create_growth_analyst(self) -> Agent:
        """Create the Growth & Revenue Analyst agent"""
        return Agent(
            role='Growth & Revenue Analyst',
            goal='Analyze revenue growth trends, KPIs, and pricing power to assess business momentum',
            backstory="""You are a quantitative analyst specializing in growth metrics 
            and revenue analysis. You have a strong background in statistics and financial 
            modeling. You calculate and interpret growth rates (QoQ, YoY, CAGR), analyze 
            key performance indicators specific to different business models, and assess 
            pricing power through margin analysis. You can identify whether growth is 
            organic or inorganic, sustainable or one-time, and evaluate the quality of 
            revenue. You understand metrics like customer acquisition cost (CAC), 
            lifetime value (LTV), retention rates, and other industry-specific KPIs.""",
            llm=self.analysis_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )
    
    def create_valuation_specialist(self) -> Agent:
        """Create the Valuation Specialist agent"""
        return Agent(
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
            and highlighting key value drivers.""",
            llm=self.analysis_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )
    
    def create_investment_advisor(self) -> Agent:
        """Create the Senior Investment Advisor agent"""
        return Agent(
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
            future potential and downside protection.""",
            llm=self.analysis_llm,
            verbose=True,
            allow_delegation=False,
            max_iter=5
        )
    
    def create_agents(self) -> List[Agent]:
        """Create and return all financial analysis agents in execution order"""
        
        agents = [
            self.create_document_analyst(),
            self.create_business_analyst(),
            self.create_growth_analyst(),
            self.create_valuation_specialist(),
            self.create_investment_advisor()
        ]
        
        return agents