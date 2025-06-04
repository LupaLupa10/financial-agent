import os
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add project directory to Python path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from financial_agent import TradingAgent
except ImportError as e:
    print(f"Import Error: {e}")
    print("Python Path:")
    for path in sys.path:
        print(f"  {path}")
    sys.exit(1)

def main():
    # Load environment variables
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables")
        return
    
    try:
        # Initialize agent
        agent = TradingAgent(openai_api_key=api_key)
        
        # Analyze a stock
        ticker = "AAPL"
        print(f"\nAnalyzing {ticker}...")
        
        analysis = agent.analyze(ticker)
        print("\nAnalysis Results:")
        for key, value in analysis.items():
            print(f"\n{key.upper()}:")
            print(value)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    
