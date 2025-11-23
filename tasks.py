"""
Financial Analysis Tasks
Defines all tasks for the financial analysis workflow, adapted for LangChain
"""
from typing import List, Dict, Any

class FinancialTasks:
    """Factory class for creating financial analysis tasks"""

    def __init__(self):
        """Initialize the tasks factory"""
        pass

    def create_document_extraction_task(self, extracted_docs: Dict[str, str], company_name: str) -> Dict[str, Any]:
        """Create task for extracting and organizing financial data"""

        docs_context = self._format_documents(extracted_docs)

        return {
            "input": f"""Analyze the following financial documents for {company_name} and extract
            all relevant financial data in a comprehensive, organized manner.

            Extract and organize:

            1. COMPANY INFORMATION
               - Company name and ticker symbol (if available)
               - Fiscal year and reporting periods
               - Industry/sector

            2. INCOME STATEMENT DATA
               - Revenue (by quarter/year, with growth rates)
               - Cost of revenue / Cost of goods sold
               - Gross profit and gross margin %
               - Operating expenses (breakdown if available)
               - Operating income and operating margin %
               - Net income and net margin %
               - Earnings per share (EPS)

            3. BALANCE SHEET DATA
               - Total assets
               - Current assets (cash, receivables, inventory)
               - Total liabilities
               - Current liabilities
               - Shareholders' equity
               - Key ratios (current ratio, debt-to-equity, etc.)

            4. CASH FLOW DATA
               - Operating cash flow
               - Investing cash flow
               - Financing cash flow
               - Free cash flow
               - Capital expenditures

            5. KEY METRICS & RATIOS
               - Return on equity (ROE)
               - Return on assets (ROA)
               - Any company-specific KPIs mentioned

            Financial Documents:
            {docs_context}

            Provide a STRUCTURED, DETAILED summary with all numbers clearly labeled with
            their reporting periods. If data is missing or unclear, note that explicitly.
            Calculate any obvious growth rates or trends you observe.

            Expected output:
            Comprehensive structured financial data extraction organized by:
            - Company information section
            - Income statement metrics with trends
            - Balance sheet snapshot with key ratios
            - Cash flow analysis
            - Calculated growth rates and financial health indicators
            All data clearly labeled with periods and units."""
        }

    def create_business_analysis_task(self, extracted_docs: Dict[str, str], company_name: str, context: str) -> Dict[str, Any]:
        """Create task for analyzing business model"""

        docs_context = self._format_documents(extracted_docs)

        return {
            "input": f"""Based on the financial documents and any business information
            contained within them, and the following context, provide a comprehensive analysis of {company_name}'s
            business model and operations.

            Previous analysis context:
            {context}

            Analyze and explain:

            1. BUSINESS OVERVIEW
               - What does the company do? (core products/services)
               - What problem does it solve for customers?
               - Brief company history/background if available

            2. REVENUE MODEL
               - How does the company make money?
               - What are the primary revenue streams?
               - Is it B2B, B2C, or both?
               - Revenue model type (subscription, transactional, licensing, etc.)
               - Geographic revenue breakdown if available

            3. CUSTOMER BASE
               - Who are the target customers?
               - What customer segments does it serve?
               - Any information on customer concentration or diversity

            4. COMPETITIVE POSITIONING
               - What is the company's competitive advantage (moat)?
               - Market position (leader, challenger, niche player?)
               - Any mentioned competitive threats or advantages

            5. BUSINESS QUALITY ASSESSMENT
               - Business model sustainability
               - Scalability potential
               - Cyclicality vs. recurring revenue
               - Any regulatory or market risks mentioned

            Financial Documents:
            {docs_context}

            Synthesize information from the documents and provide clear explanations.
            If certain information isn't available in the documents, note that and make
            reasonable inferences based on the financial data patterns.

            Expected output:
            Clear, comprehensive business model analysis including:
            - Plain-language explanation of what the company does
            - Detailed revenue model breakdown
            - Customer base and market positioning
            - Competitive advantages and moats
            - Business quality assessment with specific supporting evidence from documents"""
        }

    def create_growth_analysis_task(self, extracted_docs: Dict[str, str], company_name: str, context: str) -> Dict[str, Any]:
        """Create task for analyzing growth metrics"""

        docs_context = self._format_documents(extracted_docs)

        return {
            "input": f"""Conduct a thorough analysis of {company_name}'s growth
            trajectory, key performance indicators, and pricing power, based on the provided context.

            Previous analysis context:
            {context}

            Analyze the following:

            1. REVENUE GROWTH ANALYSIS
               - Calculate quarter-over-quarter (QoQ) growth rates
               - Calculate year-over-year (YoY) growth rates
               - Calculate CAGR if multiple years available
               - Identify acceleration or deceleration trends
               - Compare to industry benchmarks if mentioned

            2. PROFITABILITY TRENDS
               - Gross margin trends over time
               - Operating margin trends over time
               - Net margin trends over time
               - Are margins expanding or contracting?

            3. KEY PERFORMANCE INDICATORS
               - Identify all company-specific KPIs mentioned
               - Analyze KPI trends (customer growth, retention, etc.)
               - Evaluate KPI health and trajectory
               - Industry-standard metrics if applicable

            4. PRICING POWER ASSESSMENT
               - Evidence of pricing power in margin trends
               - Revenue growth vs. volume growth indicators
               - Premium pricing vs. commodity pricing
               - Ability to pass costs to customers

            5. GROWTH QUALITY & SUSTAINABILITY
               - Is growth organic or acquisition-driven?
               - Revenue quality (recurring vs. one-time)
               - Cash generation vs. accounting profits
               - Sustainability of current growth rates

            Financial Documents:
            {docs_context}

            Provide specific calculations with percentages and clearly show your work.
            Identify both positive momentum and concerning trends.

            Expected output:
            Detailed growth and KPI analysis including:
            - Specific growth rate calculations (QoQ, YoY, CAGR) with numbers
            - Margin trend analysis with data points
            - Complete KPI assessment with trajectories
            - Clear pricing power evaluation with evidence
            - Growth quality assessment with supporting metrics"""
        }

    def create_valuation_task(self, extracted_docs: Dict[str, str],
                            valuation_params: str, company_name: str, context: str) -> Dict[str, Any]:
        """Create task for company valuation"""

        docs_context = self._format_documents(extracted_docs)

        return {
            "input": f"""Perform a comprehensive valuation analysis of {company_name}
            using the provided valuation parameters, methodologies, and context.

            Previous analysis context:
            {context}

            Your analysis should include:

            1. MULTIPLE-BASED VALUATION
               - Calculate relevant multiples (P/E, P/S, EV/EBITDA, P/B, etc.)
               - Compare to industry averages/ranges from parameters
               - Determine if multiples suggest overvaluation or undervaluation

            2. INTRINSIC VALUE CALCULATION
               - Apply DCF or other intrinsic value methods from parameters
               - Use appropriate discount rates from parameters
               - Make reasonable growth assumptions based on historical data
               - Calculate fair value estimate

            3. COMPARABLE ANALYSIS
               - Compare to peer companies if benchmarks provided
               - Adjust for size, growth, and profitability differences
               - Determine relative valuation

            4. VALUATION RANGE
               - Synthesize multiple approaches
               - Provide bear, base, and bull case valuations
               - Explain key assumptions and sensitivities

            5. VALUATION OPINION
               - Is the stock overvalued, fairly valued, or undervalued?
               - What is the implied upside/downside?
               - Key value drivers and risks to valuation

            Valuation Parameters and Methodologies:
            {valuation_params}

            Financial Data:
            {docs_context}

            Show all calculations clearly. Use the methodologies and parameters provided.
            Be explicit about assumptions and their impact on valuation.

            Expected output:
            Comprehensive valuation analysis with:
            - Multiple valuation approaches with calculations shown
            - Clear fair value estimate or range
            - Comparison to current price (if available)
            - Bear/base/bull scenarios
            - Key assumptions and sensitivities clearly stated
            - Final valuation opinion (overvalued/fairly valued/undervalued)"""
        }

    def create_investment_recommendation_task(self, company_name: str, context: str) -> Dict[str, Any]:
        """Create final investment recommendation task"""

        return {
            "input": f"""As a Senior Investment Advisor, synthesize all previous
            analyses to provide a clear, actionable investment recommendation for {company_name}.

            Previous analysis context:
            {context}

            Your recommendation must include:

            1. INVESTMENT RATING
               - Clear rating: STRONG BUY, BUY, HOLD, SELL, or STRONG SELL
               - Conviction level: High, Medium, or Low

            2. INVESTMENT THESIS (3-5 key points)
               - Why this is a good/bad investment opportunity
               - Key strengths that support the thesis
               - What makes this company attractive or unattractive

            3. SUPPORTING EVIDENCE
               - Financial health summary
               - Business model strength
               - Growth prospects and sustainability
               - Valuation attractiveness
               - Competitive positioning

            4. KEY RISKS (Top 3-5)
               - What could go wrong?
               - Execution risks
               - Market/competitive risks
               - Financial risks
               - Regulatory or other external risks

            5. VALUATION PERSPECTIVE
               - Current valuation assessment
               - Target price or price range (if applicable)
               - Expected return potential
               - Time horizon considerations

            6. WHO SHOULD INVEST
               - Investor profile suited for this stock
               - Risk tolerance needed
               - Investment time horizon

            7. FINAL VERDICT
               - One paragraph summary of your recommendation
               - Clear action item for the reader

            Be honest, balanced, and specific. Consider both bull and bear cases.
            Make your recommendation actionable and clear. Back up opinions with
            evidence from the analyses.

            Expected output:
            Clear, actionable investment recommendation including:
            - Explicit rating (Strong Buy/Buy/Hold/Sell/Strong Sell)
            - Concise investment thesis (3-5 bullet points)
            - Supporting evidence from all analyses
            - Top risks clearly identified
            - Valuation perspective and target
            - Investor suitability assessment
            - Final verdict paragraph with clear action
            Written in professional yet accessible language suitable for an investment decision."""
        }
    
    def create_tasks(self, extracted_docs: Dict[str, str], 
                    valuation_params: str, company_name: str) -> List[Dict[str, Any]]:
        """Create all tasks in the proper sequence"""
        
        # In this new setup, we create the tasks as dictionaries of inputs for the agents.
        # The context from previous tasks will be manually passed in the orchestrator.
        
        task_extract = self.create_document_extraction_task(
            extracted_docs=extracted_docs,
            company_name=company_name
        )
        
        # The subsequent tasks will need context from the previous ones.
        # We will create placeholders for the context, which will be filled in the orchestrator.
        task_business = self.create_business_analysis_task(
            extracted_docs=extracted_docs,
            company_name=company_name,
            context="{context_placeholder}"
        )
        
        task_growth = self.create_growth_analysis_task(
            extracted_docs=extracted_docs,
            company_name=company_name,
            context="{context_placeholder}"
        )
        
        task_valuation = self.create_valuation_task(
            extracted_docs=extracted_docs,
            valuation_params=valuation_params,
            company_name=company_name,
            context="{context_placeholder}"
        )
        
        task_recommendation = self.create_investment_recommendation_task(
            company_name=company_name,
            context="{context_placeholder}"
        )
        
        return [task_extract, task_business, task_growth, task_valuation, task_recommendation]

    @staticmethod
    def _format_documents(extracted_docs: Dict[str, str]) -> str:
        """Format extracted documents for inclusion in task descriptions"""

        formatted = []
        for filename, content in extracted_docs.items():
            formatted.append(f"\n{'='*70}")
            formatted.append(f"DOCUMENT: {filename}")
            formatted.append(f"{'='*70}")
            formatted.append(content)

        return "\n".join(formatted)