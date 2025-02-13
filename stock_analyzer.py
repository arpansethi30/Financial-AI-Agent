# stock_analyzer.py
import os
import logging
from datetime import datetime, timedelta, date
import pandas as pd
import yfinance as yf
from typing import Dict, Optional, List, Any, Tuple, Set
from dataclasses import dataclass, field
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq
import numpy as np
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class FundamentalMetrics:
    """Comprehensive fundamental metrics for stock analysis"""
    # Valuation Metrics
    market_cap: float
    enterprise_value: float
    pe_ratio: float
    forward_pe: float
    pb_ratio: float
    ev_to_ebitda: float
    
    # Financial Health Metrics
    debt_to_equity: float
    current_ratio: float
    quick_ratio: float
    interest_coverage: float
    
    # Profitability Metrics
    roe: float
    roa: float
    roic: float
    gross_margin: float
    operating_margin: float
    net_margin: float
    
    # Growth & Income Metrics
    revenue_growth: float
    earnings_growth: float
    free_cash_flow_growth: float
    dividend_yield: float
    dividend_payout_ratio: float

@dataclass
class QualitativeMetrics:
    """Qualitative metrics based on Buffett's principles"""
    business_understanding: str
    competitive_advantage: str
    management_quality: str
    sector_outlook: str
    brand_value: str
    pricing_power: str
    market_share: str
    regulatory_risks: str

@dataclass
class MoatAnalysis:
    """Detailed analysis of company's economic moat"""
    brand_power: float
    network_effects: float
    switching_costs: float
    cost_advantages: float
    intangible_assets: float
    market_share: float
    overall_score: float = field(init=False)
    
    def __post_init__(self):
        weights = [0.25, 0.2, 0.15, 0.2, 0.1, 0.1]
        scores = [self.brand_power, self.network_effects, self.switching_costs,
                 self.cost_advantages, self.intangible_assets, self.market_share]
        self.overall_score = sum(w * s for w, s in zip(weights, scores))

@dataclass
class ManagementAnalysis:
    """Analysis of management quality and integrity"""
    capital_allocation_score: float
    insider_ownership: float
    accounting_quality: float
    debt_management: float
    shareholder_friendly: float
    execution_track_record: float

@dataclass
class SectorComparison:
    """Sector-specific comparison metrics"""
    sector_pe: float
    sector_pb: float
    sector_roe: float
    sector_median_mcap: float
    relative_valuation: float
    peer_companies: List[str]
    peer_metrics: Dict[str, Dict[float, float]]
    valuation_percentile: float

class FinancialDataFetcher:
    """Handles all data fetching operations with robust error handling"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = timedelta(hours=4)
        
    def _is_cache_valid(self, ticker: str) -> bool:
        if ticker not in self.cache:
            return False
        cache_time = self.cache[ticker]["timestamp"]
        return datetime.now() - cache_time < self.cache_duration
    
    def get_stock_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch stock data with caching and error handling"""
        try:
            if self._is_cache_valid(ticker):
                logger.info(f"Using cached data for {ticker}")
                return self.cache[ticker]["data"]
            
            logger.info(f"Fetching fresh data for {ticker}")
            stock = yf.Ticker(ticker + ".NS")
            info = stock.info
            
            # Get additional financial data
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cash_flow
            
            # Get historical data for trend analysis
            hist_data = stock.history(period="5y")
            
            # Get quarterly results for more granular analysis
            quarterly_financials = stock.quarterly_financials
            quarterly_balance_sheet = stock.quarterly_balance_sheet
            quarterly_cashflow = stock.quarterly_cashflow
            
            data = {
                "info": info,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "historical_data": hist_data,
                "quarterly_financials": quarterly_financials,
                "quarterly_balance_sheet": quarterly_balance_sheet,
                "quarterly_cashflow": quarterly_cashflow
            }
            
            self.cache[ticker] = {
                "timestamp": datetime.now(),
                "data": data
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching Indian stock data for {ticker}: {str(e)}")
            return None

class BuffettAnalyzer:
    """Enhanced stock analyzer based on Warren Buffett's principles"""
    
    def __init__(self):
        self.data_fetcher = FinancialDataFetcher()
        self.criteria = {
            "market_cap_min": 5000000000,  # 500 crores minimum
            "pe_max": 25,
            "pb_max": 3,
            "roe_min": 0.15,
            "debt_equity_max": 0.5,
            "current_ratio_min": 1.5,
            "operating_margin_min": 0.15,
            "revenue_growth_min": 0.10,
            "net_margin_min": 0.10,
            "fcf_positive_years": 8,  # Consistent FCF generation
            "debt_coverage_min": 3,    # Times interest earned
            "profit_growth_min": 0.12  # Minimum 12% annual growth
        }
        self.criteria.update({
            "retained_earnings_roic": 0.12,  # 12% minimum return on retained earnings
            "dividend_consistency": 10,  # years of consistent dividends
            "cash_flow_coverage": 1.5,  # operating cash flow / capital expenditure
            "margin_stability": 0.05,  # maximum standard deviation in margins
            "market_share_min": 0.15,  # minimum market share in industry
            "insider_ownership_min": 0.05  # minimum insider ownership
        })
        # Add Indian market specific criteria
        self.indian_criteria = {
            "promoter_holding_min": 0.40,  # Minimum 40% promoter holding
            "pledge_share_max": 0.20,  # Maximum 20% pledged shares
            "india_market_share": 0.05,  # Minimum 5% market share in India
            "export_revenue_max": 0.70,  # Maximum 70% export dependency
            "group_company_exposure": 0.30,  # Maximum 30% related party transactions
            "govt_dependency_max": 0.40,  # Maximum 40% government dependency
            "working_capital_days": 90,  # Maximum working capital cycle
            "india_specific_sectors": {
                "high_growth": ["IT", "Pharma", "FMCG", "Insurance"],
                "cyclical": ["Auto", "Metals", "Banking"],
                "regulated": ["Utilities", "Telecom"]
            }
        }
    
    def _calculate_growth_rates(self, data: Dict) -> Dict[str, float]:
        """Calculate various growth rates"""
        try:
            financials = data["financials"]
            if (financials.empty):
                return {}
            
            # Calculate year-over-year growth rates
            growth_rates = {}
            for metric in ["Revenue", "Net Income", "Operating Income"]:
                if metric in financials.index:
                    values = financials.loc[metric]
                    yoy_growth = (values.iloc[0] - values.iloc[1]) / abs(values.iloc[1])
                    growth_rates[f"{metric.lower()}_growth"] = yoy_growth
            
            return growth_rates
            
        except Exception as e:
            logger.error(f"Error calculating growth rates: {str(e)}")
            return {}

    def _extract_fundamentals(self, data: Dict) -> Optional[FundamentalMetrics]:
        """Extract and calculate fundamental metrics"""
        try:
            info = data["info"]
            growth_rates = self._calculate_growth_rates(data)
            
            return FundamentalMetrics(
                market_cap=info.get("marketCap", 0),
                enterprise_value=info.get("enterpriseValue", 0),
                pe_ratio=info.get("forwardPE", float("inf")),
                forward_pe=info.get("forwardPE", float("inf")),
                pb_ratio=info.get("priceToBook", float("inf")),
                ev_to_ebitda=info.get("enterpriseToEbitda", float("inf")),
                debt_to_equity=info.get("debtToEquity", float("inf")) / 100 if info.get("debtToEquity") else float("inf"),
                current_ratio=info.get("currentRatio", 0),
                quick_ratio=info.get("quickRatio", 0),
                interest_coverage=info.get("interestCoverage", 0),
                roe=info.get("returnOnEquity", 0),
                roa=info.get("returnOnAssets", 0),
                roic=info.get("returnOnCapital", 0),
                gross_margin=info.get("grossMargins", 0),
                operating_margin=info.get("operatingMargins", 0),
                net_margin=info.get("profitMargins", 0),
                revenue_growth=growth_rates.get("revenue_growth", 0),
                earnings_growth=growth_rates.get("net_income_growth", 0),
                free_cash_flow_growth=growth_rates.get("operating_income_growth", 0),
                dividend_yield=info.get("dividendYield", 0) if info.get("dividendYield") else 0,
                dividend_payout_ratio=info.get("payoutRatio", 0) if info.get("payoutRatio") else 0
            )
            
        except Exception as e:
            logger.error(f"Error extracting fundamentals: {str(e)}")
            return None

    def _analyze_competitive_advantage(self, data: Dict) -> str:
        """Analyze competitive advantage (Moat)"""
        info = data["info"]
        financials = data["financials"]
        
        moat_indicators = []
        
        # Check for consistent high margins
        if info.get("operatingMargins", 0) > 0.20:
            moat_indicators.append("High operating margins indicate pricing power")
            
        # Check for consistent ROE
        if info.get("returnOnEquity", 0) > 0.20:
            moat_indicators.append("High ROE indicates strong competitive position")
            
        # Market leader check
        if info.get("marketCap", 0) > 100000000000:  # 10,000 crores
            moat_indicators.append("Significant market presence")
            
        return " | ".join(moat_indicators) if moat_indicators else "No significant moat identified"

    def _analyze_management_quality(self, data: Dict) -> ManagementAnalysis:
        """Enhanced analysis of management quality"""
        info = data["info"]
        financials = data["financials"]
        
        # Analyze capital allocation
        roic = info.get("returnOnCapital", 0)
        capital_allocation = min(1.0, roic / 0.2)  # Scale by 20% ROIC
        
        # Insider ownership analysis
        insider_ownership = min(1.0, info.get("heldPercentInsiders", 0) / 100)
        
        # Accounting quality (based on accruals and cash flow consistency)
        accounting_quality = 0.8  # Default to conservative estimate
        
        # Debt management
        debt_to_equity = info.get("debtToEquity", 0)
        debt_management = 1.0 - min(1.0, debt_to_equity / 100)
        
        # Shareholder friendly policies
        dividend_payout = info.get("payoutRatio", 0)
        shareholder_friendly = min(1.0, dividend_payout / 0.75)
        
        # Track record of execution
        execution_track = min(1.0, info.get("returnOnEquity", 0) / 0.2)
        
        return ManagementAnalysis(
            capital_allocation_score=capital_allocation,
            insider_ownership=insider_ownership,
            accounting_quality=accounting_quality,
            debt_management=debt_management,
            shareholder_friendly=shareholder_friendly,
            execution_track_record=execution_track
        )

    def _analyze_circle_of_competence(self, data: Dict) -> str:
        """Analyze if business falls within Buffett's circle of competence"""
        info = data["info"]
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        
        buffett_sectors = {
            "Consumer Staples": "Strong match - Buffett favors consumer staples",
            "Financials": "Strong match - Buffett understands banking and insurance well",
            "Consumer Discretionary": "Moderate match - Depends on brand strength",
            "Industrials": "Moderate match - Requires deep industry knowledge",
            "Technology": "Weak match - Generally outside traditional circle of competence",
            "Healthcare": "Moderate match - Favors established companies with strong moats"
        }
        
        return buffett_sectors.get(sector, "Outside circle of competence - Requires careful analysis")

    def _analyze_moat_strength(self, data: Dict) -> MoatAnalysis:
        """Detailed analysis of company's economic moat"""
        info = data["info"]
        financials = data["financials"]
        
        # Calculate brand power from market position and margins
        brand_power = min(1.0, info.get("operatingMargins", 0) * 2)
        
        # Network effects from market share and growth
        network_effects = min(1.0, info.get("marketCap", 0) / 1e12)  # Scale by trillion
        
        # Switching costs from recurring revenue and customer retention
        switching_costs = 0.7 if info.get("sector") in ["Technology", "Financials"] else 0.3
        
        # Cost advantages from margins and scale
        cost_advantages = min(1.0, info.get("grossMargins", 0))
        
        # Intangible assets (patents, licenses, etc.)
        intangible_assets = 0.8 if info.get("sector") in ["Healthcare", "Technology"] else 0.4
        
        # Market share in industry
        market_share = min(1.0, info.get("marketCap", 0) / 1e12)
        
        return MoatAnalysis(
            brand_power=brand_power,
            network_effects=network_effects,
            switching_costs=switching_costs,
            cost_advantages=cost_advantages,
            intangible_assets=intangible_assets,
            market_share=market_share
        )

    def _analyze_historical_performance(self, data: Dict) -> Dict[str, float]:
        """Analyze long-term historical performance"""
        try:
            hist_data = data["historical_data"]
            if hist_data.empty:
                return {
                    "10y_cagr": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0
                }
            
            # Calculate returns using iloc for position-based indexing
            returns = hist_data["Close"].pct_change()
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate CAGR using iloc
            first_price = hist_data["Close"].iloc[0]
            last_price = hist_data["Close"].iloc[-1]
            years = len(hist_data.index) / 252  # Convert days to years
            cagr = (last_price / first_price) ** (1 / years) - 1 if years > 0 else 0
            
            return {
                "10y_cagr": cagr,
                "volatility": volatility,
                "sharpe_ratio": (cagr - 0.05) / volatility if volatility > 0 else 0,
                "max_drawdown": self._calculate_max_drawdown(hist_data["Close"])
            }
        except Exception as e:
            logger.error(f"Error calculating historical performance: {str(e)}")
            return {
                "10y_cagr": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0
            }

    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        rolling_max = prices.expanding(min_periods=1).max()
        drawdowns = prices / rolling_max - 1.0
        return drawdowns.min()

    def analyze_stock(self, ticker: str) -> Dict:
        """Perform comprehensive stock analysis"""
        logger.info(f"Starting analysis for {ticker}")
        
        data = self.data_fetcher.get_stock_data(ticker)
        if not data:
            return {
                "status": "error",
                "message": f"Could not fetch data for {ticker}",
                "timestamp": datetime.now().isoformat()
            }
            
        try:
            metrics = self._extract_fundamentals(data)
            qualitative_metrics = self._analyze_qualitative_factors(data)
            
            analysis = {
                "status": "success",
                "ticker": ticker,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics.__dict__ if metrics else {},
                "qualitative_metrics": qualitative_metrics.__dict__ if qualitative_metrics else {},
                "competitive_advantage": self._analyze_competitive_advantage(data),
                "management_quality": self._analyze_management_quality(data),
                "evaluation": self._evaluate_metrics(metrics) if metrics else {},
                "recommendation": None
            }
            
            analysis["recommendation"] = self._generate_buffett_recommendation(analysis)
            analysis.update({
                "circle_of_competence": self._analyze_circle_of_competence(data),
                "moat_analysis": self._analyze_moat_strength(data).__dict__,
                "management_analysis": self._analyze_management_quality(data).__dict__,
                "historical_performance": self._analyze_historical_performance(data)
            })
            # Add new analyses
            sector_comparison = self._analyze_sector_metrics(ticker, data)
            indian_specifics = self._analyze_indian_market_specifics(data)
            
            analysis.update({
                "sector_comparison": sector_comparison.__dict__ if sector_comparison else {},
                "indian_market_specifics": indian_specifics,
                "indian_criteria_evaluation": self._evaluate_indian_criteria(indian_specifics)
            })
            
            # Update recommendation to include Indian market factors
            analysis["recommendation"] = self._generate_enhanced_recommendation(analysis)
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            return {
                "status": "error",
                "message": f"Error analyzing {ticker}: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def _evaluate_metrics(self, metrics: FundamentalMetrics) -> Dict[str, bool]:
        """Evaluate each metric against Buffett's criteria"""
        evaluation = {
            "market_cap": metrics.market_cap >= self.criteria["market_cap_min"],
            "pe_ratio": metrics.pe_ratio <= self.criteria["pe_max"],
            "roe": metrics.roe >= self.criteria["roe_min"],
            "debt_equity": metrics.debt_to_equity <= self.criteria["debt_equity_max"],
            "operating_margin": metrics.operating_margin >= self.criteria["operating_margin_min"],
            "revenue_growth": metrics.revenue_growth >= self.criteria["revenue_growth_min"],
            "net_margin": metrics.net_margin >= self.criteria["net_margin_min"]
        }
        return evaluation
    
    def _generate_recommendation(self, evaluation: Dict[str, bool]) -> str:
        """Generate detailed investment recommendation"""
        failed_criteria = [k for k, v in evaluation.items() if not v]
        
        if not failed_criteria:
            return "STRONG BUY - Meets all Warren Buffett's criteria"
        elif len(failed_criteria) <= 2:
            return f"HOLD - Generally good but fails {', '.join(failed_criteria)}"
        else:
            return f"PASS - Does not meet enough criteria: fails {', '.join(failed_criteria)}"

    def _analyze_qualitative_factors(self, data: Dict) -> QualitativeMetrics:
        """Analyze qualitative factors based on Buffett's principles"""
        info = data["info"]
        
        return QualitativeMetrics(
            business_understanding="Simple and understandable business" if info.get("industry") in ["Consumer Goods", "Banking", "Insurance"] else "Complex business model",
            competitive_advantage=self._analyze_competitive_advantage(data),
            management_quality=self._analyze_management_quality(data),
            sector_outlook=self._analyze_sector_outlook(data),
            brand_value="Strong brand" if info.get("marketCap", 0) > 100000000000 else "Limited brand value",
            pricing_power="High" if info.get("operatingMargins", 0) > 0.20 else "Limited",
            market_share="Leader" if info.get("marketCap", 0) > 100000000000 else "Challenger",
            regulatory_risks="Low" if info.get("industry") not in ["Banking", "Pharma"] else "High"
        )

    def _generate_buffett_recommendation(self, analysis: Dict) -> str:
        """Generate recommendation based on Buffett's principles"""
        metrics = analysis.get("metrics", {})
        qual_metrics = analysis.get("qualitative_metrics", {})
        
        strong_points = []
        weak_points = []
        
        # Check fundamental criteria
        if metrics.get("roe", 0) > 0.15:
            strong_points.append("High ROE")
        else:
            weak_points.append("Low ROE")
            
        if metrics.get("debt_to_equity", float("inf")) < 0.5:
            strong_points.append("Low debt")
        else:
            weak_points.append("High debt")
            
        # Check qualitative criteria
        if isinstance(qual_metrics, dict):
            if qual_metrics.get("brand_value", "").startswith("Strong"):
                strong_points.append("Strong brand value")
            
            if qual_metrics.get("pricing_power", "").startswith("High"):
                strong_points.append("Good pricing power")
        elif isinstance(qual_metrics, QualitativeMetrics):
            if qual_metrics.brand_value.startswith("Strong"):
                strong_points.append("Strong brand value")
            
            if qual_metrics.pricing_power.startswith("High"):
                strong_points.append("Good pricing power")
            
        # Generate recommendation
        if len(strong_points) >= 3 and len(weak_points) <= 1:
            return f"STRONG BUY - Matches Buffett's Criteria: {', '.join(strong_points)}"
        elif len(strong_points) >= 2:
            return f"CONSIDER - Some Strength: {', '.join(strong_points)}, but concerns: {', '.join(weak_points)}"
        else:
            return f"PASS - Does not match Buffett's criteria. Concerns: {', '.join(weak_points)}"

    def _analyze_sector_outlook(self, data: Dict) -> str:
        """Analyze sector outlook"""
        info = data["info"]
        sector = info.get("sector", "Unknown")
        
        # You could enhance this with actual sector analysis data
        growing_sectors = ["Technology", "Healthcare", "Consumer Staples"]
        return "Positive" if sector in growing_sectors else "Neutral"

    def _get_sector_peers(self, ticker: str, sector: str) -> List[str]:
        """Get list of peer companies in the same sector"""
        # Define major Indian sector peers
        sector_peers = {
            "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"],
            "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS"],
            "FMCG": ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS"],
            # Add more sectors as needed
        }
        return sector_peers.get(sector, [])

    def _analyze_sector_metrics(self, ticker: str, data: Dict) -> SectorComparison:
        """Compare company with sector peers"""
        try:
            info = data["info"]
            sector = info.get("sector", "Unknown")
            peers = self._get_sector_peers(ticker, sector)
            
            peer_metrics = {}
            for peer in peers:
                peer_data = self.data_fetcher.get_stock_data(peer.replace(".NS", ""))
                if peer_data:
                    peer_metrics[peer] = {
                        "pe": peer_data["info"].get("forwardPE", 0),
                        "pb": peer_data["info"].get("priceToBook", 0),
                        "roe": peer_data["info"].get("returnOnEquity", 0),
                        "market_cap": peer_data["info"].get("marketCap", 0)
                    }
            
            # Calculate sector averages
            sector_pe = np.median([m["pe"] for m in peer_metrics.values() if m["pe"] > 0])
            sector_pb = np.median([m["pb"] for m in peer_metrics.values() if m["pb"] > 0])
            sector_roe = np.median([m["roe"] for m in peer_metrics.values() if m["roe"] > 0])
            sector_mcap = np.median([m["market_cap"] for m in peer_metrics.values()])
            
            # Calculate relative valuation
            company_pe = info.get("forwardPE", 0)
            relative_val = company_pe / sector_pe if sector_pe > 0 else 1
            
            # Calculate percentile ranking
            all_pes = [m["pe"] for m in peer_metrics.values() if m["pe"] > 0]
            percentile = stats.percentileofscore(all_pes, company_pe) if all_pes else 50
            
            return SectorComparison(
                sector_pe=sector_pe,
                sector_pb=sector_pb,
                sector_roe=sector_roe,
                sector_median_mcap=sector_mcap,
                relative_valuation=relative_val,
                peer_companies=list(peer_metrics.keys()),
                peer_metrics=peer_metrics,
                valuation_percentile=percentile
            )
            
        except Exception as e:
            logger.error(f"Error in sector analysis: {str(e)}")
            return None

    def _analyze_indian_market_specifics(self, data: Dict) -> Dict[str, Any]:
        """Analyze India-specific factors"""
        info = data["info"]
        
        return {
            "promoter_holding": info.get("heldPercentInsiders", 0),
            "pledged_shares": info.get("sharesPercentSharesOut", 0),
            "domestic_revenue": 1 - info.get("internationalRevenue", 0),
            "working_capital_efficiency": self._calculate_working_capital_cycle(data),
            "govt_dependency": self._analyze_government_dependency(data),
            "group_company_exposure": self._analyze_related_party_exposure(data)
        }

    def _evaluate_indian_criteria(self, metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate against Indian market criteria"""
        return {
            "promoter_holding": metrics["promoter_holding"] >= self.indian_criteria["promoter_holding_min"],
            "pledged_shares": metrics["pledged_shares"] <= self.indian_criteria["pledge_share_max"],
            "domestic_market": metrics["domestic_revenue"] >= (1 - self.indian_criteria["export_revenue_max"]),
            "working_capital": metrics["working_capital_efficiency"] <= self.indian_criteria["working_capital_days"],
            "govt_exposure": metrics["govt_dependency"] <= self.indian_criteria["govt_dependency_max"],
            "group_exposure": metrics["group_company_exposure"] <= self.indian_criteria["group_company_exposure"]
        }

    def _generate_enhanced_recommendation(self, analysis: Dict) -> str:
        """Generate comprehensive recommendation including Indian market factors"""
        basic_rec = self._generate_buffett_recommendation(analysis)
        indian_eval = analysis.get("indian_criteria_evaluation", {})
        sector_comp = analysis.get("sector_comparison", {})
        
        # Additional factors to consider
        indian_strengths = [k for k, v in indian_eval.items() if v]
        indian_weaknesses = [k for k, v in indian_eval.items() if not v]
        
        # Valuation check
        relative_val = sector_comp.get("relative_valuation", 1)
        valuation_comment = (
            "Undervalued compared to peers" if relative_val < 0.8
            else "Overvalued compared to peers" if relative_val > 1.2
            else "Fairly valued compared to peers"
        )
        
        # Generate final recommendation
        recommendation = [basic_rec]
        recommendation.append(f"\nIndian Market Analysis:")
        recommendation.append(f"- {valuation_comment}")
        if indian_strengths:
            recommendation.append(f"- Strengths: {', '.join(indian_strengths)}")
        if indian_weaknesses:
            recommendation.append(f"- Concerns: {', '.join(indian_weaknesses)}")
            
        return "\n".join(recommendation)

    def _calculate_working_capital_cycle(self, data: Dict) -> float:
        """Calculate working capital cycle in days"""
        try:
            balance_sheet = data["balance_sheet"]
            financials = data["financials"]
            
            if balance_sheet.empty or financials.empty:
                return 0.0
            
            # Get latest values
            current_assets = balance_sheet.loc["Total Current Assets"].iloc[0]
            current_liabilities = balance_sheet.loc["Total Current Liabilities"].iloc[0]
            inventory = balance_sheet.loc["Inventory"].iloc[0] if "Inventory" in balance_sheet.index else 0
            receivables = balance_sheet.loc["Net Receivables"].iloc[0] if "Net Receivables" in balance_sheet.index else 0
            payables = balance_sheet.loc["Accounts Payable"].iloc[0] if "Accounts Payable" in balance_sheet.index else 0
            revenue = financials.loc["Total Revenue"].iloc[0] if "Total Revenue" in financials.index else 1
            
            # Calculate working capital metrics
            working_capital = current_assets - current_liabilities
            dso = (receivables * 365) / revenue  # Days Sales Outstanding
            dio = (inventory * 365) / revenue    # Days Inventory Outstanding
            dpo = (payables * 365) / revenue     # Days Payables Outstanding
            
            # Calculate working capital cycle
            return max(0, dso + dio - dpo)
            
        except Exception as e:
            logger.error(f"Error calculating working capital cycle: {str(e)}")
            return 0.0

    def _analyze_government_dependency(self, data: Dict) -> float:
        """Analyze dependency on government contracts/regulations"""
        try:
            info = data["info"]
            sector = info.get("sector", "Unknown")
            
            # Define sectors with high government dependency
            govt_dependent_sectors = {
                "Defense": 0.8,
                "Utilities": 0.7,
                "Infrastructure": 0.6,
                "Healthcare": 0.4,
                "Banking": 0.3,
                "Telecom": 0.5
            }
            
            return govt_dependent_sectors.get(sector, 0.1)
            
        except Exception as e:
            logger.error(f"Error analyzing government dependency: {str(e)}")
            return 0.0

    def _analyze_related_party_exposure(self, data: Dict) -> float:
        """Analyze exposure to related party transactions"""
        try:
            info = data["info"]
            
            # This is a simplified analysis - in real world, would need detailed financial statements
            is_group_company = any(group in info.get("longName", "").upper() for group in [
                "TATA", "BIRLA", "RELIANCE", "ADANI", "MAHINDRA"
            ])
            
            if is_group_company:
                return 0.4  # Higher exposure for group companies
            return 0.1  # Lower exposure for standalone companies
            
        except Exception as e:
            logger.error(f"Error analyzing related party exposure: {str(e)}")
            return 0.0

def format_analysis_output(analysis: Dict) -> str:
    """Format analysis results for display"""
    if analysis["status"] == "error":
        return f"Error: {analysis['message']}"
        
    output = []
    output.append("\n" + "="*80)
    output.append(f"STOCK ANALYSIS REPORT FOR {analysis['ticker']}")
    output.append(f"Generated on: {analysis['timestamp']}")
    output.append("="*80)
    
    # Key Metrics
    output.append("\nKEY METRICS:")
    output.append("-"*80)
    metrics = analysis["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            if "ratio" in key or "margin" in key or "growth" in key or "roe" in key or "roa" in key or "roic" in key:
                formatted_value = f"{value:.2%}"
            elif "market_cap" in key or "enterprise_value" in key:
                formatted_value = f"₹{value:,.0f}"
            else:
                formatted_value = f"{value:.2f}"
        else:
            formatted_value = str(value)
        
        formatted_key = key.replace("_", " ").title()
        output.append(f"{formatted_key:<30}: {formatted_value}")
    
    # Recommendation
    output.append("\nRECOMMENDATION:")
    output.append("-"*80)
    output.append(analysis["recommendation"])
    
    output.append("\nQUALITATIVE ANALYSIS:")
    output.append("-"*80)
    for key, value in analysis.get("qualitative_metrics", {}).items():
        formatted_key = key.replace("_", " ").title()
        output.append(f"{formatted_key:<30}: {value}")
    
    output.append("\nCOMPETITIVE ADVANTAGE ANALYSIS:")
    output.append("-"*80)
    output.append(analysis.get("competitive_advantage", "Not available"))
    
    output.append("\nCIRCLE OF COMPETENCE:")
    output.append("-"*80)
    output.append(analysis.get("circle_of_competence", "Not available"))
    
    output.append("\nECONOMIC MOAT ANALYSIS:")
    output.append("-"*80)
    moat = analysis.get("moat_analysis", {})
    for key, value in moat.items():
        formatted_key = key.replace("_", " ").title()
        output.append(f"{formatted_key:<30}: {value:.2%}")
    
    output.append("\nMANAGEMENT QUALITY ANALYSIS:")
    output.append("-"*80)
    mgmt = analysis.get("management_analysis", {})
    for key, value in mgmt.items():
        formatted_key = key.replace("_", " ").title()
        output.append(f"{formatted_key:<30}: {value:.2%}")
    
    output.append("\nHISTORICAL PERFORMANCE:")
    output.append("-"*80)
    hist = analysis.get("historical_performance", {})
    for key, value in hist.items():
        formatted_key = key.replace("_", " ").title()
        output.append(f"{formatted_key:<30}: {value:.2%}")
    
    # Add sector comparison section
    output.append("\nSECTOR COMPARISON:")
    output.append("-"*80)
    sector_comp = analysis.get("sector_comparison", {})
    for key, value in sector_comp.items():
        if key != "peer_metrics":
            formatted_key = key.replace("_", " ").title()
            if isinstance(value, float):
                output.append(f"{formatted_key:<30}: {value:.2f}")
            else:
                output.append(f"{formatted_key:<30}: {value}")
    
    # Add Indian market specifics section
    output.append("\nINDIAN MARKET SPECIFICS:")
    output.append("-"*80)
    indian_metrics = analysis.get("indian_market_specifics", {})
    for key, value in indian_metrics.items():
        formatted_key = key.replace("_", " ").title()
        output.append(f"{formatted_key:<30}: {value:.2%}")
    
    return "\n".join(output)

def main():
    # Set up logging
    logging.info("Starting stock analysis program")
    
    try:
        # Initialize analyzer
        analyzer = BuffettAnalyzer()
        
        # Get stock ticker from user
        ticker = input("Enter stock ticker (e.g., INFY for Infosys): ").strip().upper()
        
        # Perform analysis
        print(f"\nAnalyzing {ticker}...")
        analysis = analyzer.analyze_stock(ticker)
        
        # Display results
        print(format_analysis_output(analysis))
        
    except Exception as e:
        logging.error(f"Program error: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()