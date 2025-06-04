from typing import Dict
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from financial_agent.tools import TradingTools


class TradingAgent:
    """Main Trading Analysis Agent"""

    def __init__(self, openai_api_key: str, alpha_vantage_key: str):
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4",
            api_key=openai_api_key
        )

        self.tools = TradingTools(api_key=alpha_vantage_key)
        self._setup_prompt()

    def _setup_prompt(self):
        """Setup the analysis prompt"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the trading data and provide comprehensive recommendations.
                         Consider technical analysis, predictions, options, and risk metrics."""),
            ("human", "{data}")
        ])

    def analyze(self, ticker: str) -> Dict:
        """Perform complete trading analysis"""
        try:
            # Use .invoke() for all tool methods
            market_data = self.tools.fetch_market_data({"ticker": ticker})
            technical = self.tools.analyze_technicals({"data": market_data})
            prediction = self.tools.predict_movement({"data": market_data})
            options = self.tools.analyze_options({"ticker": ticker})
            risk = self.tools.calculate_risk_metrics({"data": market_data})

            # Compose analysis
            analysis_data = {
                "ticker": ticker,
                "technical": technical,
                "prediction": prediction,
                "options": options,
                "risk": risk
            }

            recommendation = self.llm.invoke(
                self.prompt.format(data=str(analysis_data))
            )

            return {
                "ticker": ticker,
                "technical_analysis": technical,
                "prediction": prediction,
                "options_analysis": options,
                "risk_metrics": risk,
                "recommendation": recommendation.content
            }

        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise


if __name__ == "__main__":
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not openai_key or not alpha_key:
        raise EnvironmentError("Missing API keys in environment variables")

    agent = TradingAgent(openai_api_key=openai_key, alpha_vantage_key=alpha_key)
    result = agent.analyze("AAPL")

    print("\n=== Trading Analysis Result ===")
    for section, content in result.items():
        print(f"\n--- {section.upper()} ---")
        if isinstance(content, dict):
            for k, v in content.items():
                print(f"{k}: {v}")
        else:
            print(content)
