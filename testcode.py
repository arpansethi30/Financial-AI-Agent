import os
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional
from dataclasses import dataclass
from phi.agent import Agent
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.groq import Groq

@dataclass
class FundamentalMetrics:
    """Storage class for fundamental metrics"""
    pe_ratio: float
    pb_ratio: float
    debt_to_equity: float
    current_ratio: float
    roe: float
    roic: float
    operating_margin: float
    net_margin: float
    dividend_yield: float

class BuffettAnalyzer:
    """Analyzes stocks based on Warren Buffett's principles"""
    
    def __init__(self):
        self.min_market_cap = 500_000_000  # 500 crores minimum
        self.max_pe = 25
        self.min_roe = 0.15  # 15% minimum ROE
        self.min_operating_margin = 0.15
        self.max_debt_equity = 0.5
        self.min_current_ratio = 1.5

    def get_fundamentals(self, ticker: str) -> Optional[FundamentalMetrics]:
        """Fetch fundamental metrics for a given stock"""
        try:
            stock = yf.Ticker(ticker + ".NS")  # Adding .NS for NSE stocks
            info = stock.info
            
            # Get financial ratios
            metrics = FundamentalMetrics(
                pe_ratio=info.get('forwardPE', float('inf')),
                pb_ratio=info.get('priceToBook', float('inf')),
                debt_to_equity=info.get('debtToEquity', float('inf')) / 100 if info.get('debtToEquity') else float('inf'),
                current_ratio=info.get('currentRatio', 0),
                roe=info.get('returnOnEquity', 0),
                roic=info.get('returnOnCapital', 0),
                operating_margin=info.get('operatingMargins', 0),
                net_margin=info.get('profitMargins', 0),
                dividend_yield=info.get('dividendYield', 0) if info.get('dividendYield') else 0
            )
            return metrics
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None

    def analyze_stock(self, ticker: str) -> Dict:
        """Analyze a stock based on Buffett's principles"""
        metrics = self.get_fundamentals(ticker)
        if not metrics:
            return {"status": "error", "message": f"Could not fetch data for {ticker}"}

        analysis = {
            "ticker": ticker,
            "pass_criteria": True,
            "metrics": metrics.__dict__,
            "analysis": []
        }

        # Check each criterion
        if metrics.pe_ratio > self.max_pe:
            analysis["pass_criteria"] = False
            analysis["analysis"].append(f"PE ratio ({metrics.pe_ratio:.2f}) is above maximum threshold of {self.max_pe}")

        if metrics.roe < self.min_roe:
            analysis["pass_criteria"] = False
            analysis["analysis"].append(f"ROE ({metrics.roe:.2%}) is below minimum threshold of {self.min_roe:.2%}")

        if metrics.operating_margin < self.min_operating_margin:
            analysis["pass_criteria"] = False
            analysis["analysis"].append(f"Operating margin ({metrics.operating_margin:.2%}) is below minimum threshold of {self.min_operating_margin:.2%}")

        if metrics.debt_to_equity > self.max_debt_equity:
            analysis["pass_criteria"] = False
            analysis["analysis"].append(f"Debt/Equity ratio ({metrics.debt_to_equity:.2f}) is above maximum threshold of {self.max_debt_equity}")

        if metrics.current_ratio < self.min_current_ratio:
            analysis["pass_criteria"] = False
            analysis["analysis"].append(f"Current ratio ({metrics.current_ratio:.2f}) is below minimum threshold of {self.min_current_ratio}")

        return analysis

class IndianStockAnalyst:
    """Main class for Indian stock analysis"""
    
    def __init__(self):
        self.buffett_analyzer = BuffettAnalyzer()
        self.web_search_agent = Agent(
            name="Web Search Agent",
            role="Search the web for information",
            model=Groq(id="llama3-70b-8192"),
            tools=[DuckDuckGo()],
            instructions=["Always include sources"],
            show_tool_calls=True,
            markdown=True,
        )
        self.finance_agent = Agent(
            name="Finance Agent",
            model=Groq(id="llama3-70b-8192"),
            tools=[YFinanceTools()],
            instructions=["Use tables to display data"],
            markdown=True,
        )

    def analyze_stock(self, ticker: str) -> Dict:
        """Complete analysis of a stock"""
        try:
            # Get fundamental analysis
            fundamental_analysis = self.buffett_analyzer.analyze_stock(ticker)
            
            # Create analysis result
            analysis = {
                "fundamental_analysis": {
                    "metrics": fundamental_analysis["metrics"],
                    "criteria_met": fundamental_analysis["pass_criteria"],
                    "key_findings": fundamental_analysis["analysis"]
                },
                "recommendation": self._generate_recommendation(fundamental_analysis),
                "analysis_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Try to get news with better error handling
            try:
                news_query = f"Get latest news about {ticker} stock company updates"
                news = self.web_search_agent.run(news_query)
                analysis["recent_news"] = news if news else "No recent news found"
            except Exception as e:
                analysis["recent_news"] = "News temporarily unavailable"
            
            return analysis
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error analyzing {ticker}: {str(e)}",
                "analysis_date": datetime.now().strftime("%Y-%m-%d")
            }

    def _generate_recommendation(self, analysis: Dict) -> str:
        """Generate a more detailed recommendation"""
        if not analysis["pass_criteria"]:
            return f"FAIL - {'; '.join(analysis['analysis'])}"
        return "PASS - Meets Warren Buffett's investment criteria"

    def screen_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """Screen multiple stocks and return a DataFrame of results"""
        results = []
        for ticker in tickers:
            analysis = self.buffett_analyzer.analyze_stock(ticker)
            if "metrics" in analysis:
                results.append({
                    "Ticker": ticker,
                    "PE Ratio": f"{analysis['metrics']['pe_ratio']:.2f}",
                    "ROE": f"{analysis['metrics']['roe']:.2%}",
                    "Operating Margin": f"{analysis['metrics']['operating_margin']:.2%}",
                    "Debt/Equity": f"{analysis['metrics']['debt_to_equity']:.2f}",
                    "Current Ratio": f"{analysis['metrics']['current_ratio']:.2f}",
                    "Passed Criteria": "✓" if analysis["pass_criteria"] else "✗"
                })
        
        return pd.DataFrame(results)

# Usage example:
if __name__ == "__main__":
    # Initialize the analyst
    analyst = IndianStockAnalyst()
    
    # Example Indian stocks (NSE tickers)
    sample_tickers = ["INFY", "TCS", "HDFCBANK", "RELIANCE", "TATAMOTORS"]
    
    # Screen multiple stocks
    results_df = analyst.screen_stocks(sample_tickers)
    print("\nStock Screening Results:")
    print(results_df)
    
    # Detailed analysis of a single stock
    detailed_analysis = analyst.analyze_stock("INFY")
    print("\nDetailed Analysis for INFY:")
    print(detailed_analysis)