# run_analysis.py
from testcode import IndianStockAnalyst
import pandas as pd
import json

def format_metric(key: str, value: float) -> str:
    """Format individual metrics with appropriate formatting"""
    percentage_metrics = ['roe', 'operating_margin', 'net_margin', 'dividend_yield']
    ratio_metrics = ['pe_ratio', 'pb_ratio', 'debt_to_equity', 'current_ratio']
    
    if key in percentage_metrics:
        return f"{value:.2%}"
    elif key in ratio_metrics:
        return f"{value:.2f}"
    return str(value)

def format_analysis(analysis):
    """Format the analysis output for better readability"""
    if "status" in analysis and analysis["status"] == "error":
        return f"Error: {analysis['message']}"
    
    output = []
    output.append("\n" + "="*50)
    output.append("FUNDAMENTAL METRICS")
    output.append("="*50)
    
    metrics = analysis["fundamental_analysis"]["metrics"]
    max_key_length = max(len(key.replace('_', ' ').title()) for key in metrics.keys())
    
    for key, value in metrics.items():
        formatted_key = key.replace('_', ' ').title()
        if isinstance(value, (int, float)):
            formatted_value = format_metric(key, value)
            output.append(f"{formatted_key:<{max_key_length}} : {formatted_value}")
        else:
            output.append(f"{formatted_key:<{max_key_length}} : {value}")
    
    output.append("\n" + "="*50)
    output.append("RECOMMENDATION")
    output.append("="*50)
    output.append(analysis['recommendation'])
    
    if analysis.get("recent_news"):
        output.append("\n" + "="*50)
        output.append("RECENT NEWS")
        output.append("="*50)
        # Extract just the news content without the metadata
        news_content = analysis["recent_news"]
        if isinstance(news_content, str):
            if "content=" in news_content:
                # Extract the actual news content from the string
                news_content = news_content.split('content=')[1].split('content_type=')[0].strip("'")
            output.append(news_content)
    
    return "\n".join(output)

def main():
    # Initialize the analyst
    print("Initializing Indian Stock Analyst...")
    analyst = IndianStockAnalyst()
    
    # List of stocks to analyze
    stocks_to_analyze = ["INFY", "TCS", "HDFCBANK", "RELIANCE", "TATAMOTORS"]
    
    # Screen multiple stocks
    print("\nScreening multiple stocks...")
    results = analyst.screen_stocks(stocks_to_analyze)
    
    # Display results
    print("\nScreening Results:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.colheader_justify', 'left')
    print(results)
    
    # Get detailed analysis for a single stock
    print("\nDetailed Analysis for INFY:")
    detailed_analysis = analyst.analyze_stock("INFY")
    print(format_analysis(detailed_analysis))

if __name__ == "__main__":
    main()