from typing import Dict, Optional
from langchain_core.pydantic_v1 import BaseModel, Field


class TickerInput(BaseModel):
    """Schema for ticker input"""
    ticker: str = Field(
        ..., 
        description="Stock ticker symbol",
        example="AAPL"
    )

class DataInput(BaseModel):
    """Schema for market data input"""
    data: Dict = Field(
        ..., 
        description="Market data dictionary containing historical_data and company_info",
        example={
            "historical_data": {"Close": [100, 101, 102]},
            "company_info": {"name": "Apple Inc."}
        }
    )

class TradingState(BaseModel):
    """State management for trading analysis"""
    ticker: str
    stock_data: Optional[Dict] = None
    technical_analysis: Optional[Dict] = None
    prediction: Optional[Dict] = None
    options_analysis: Optional[Dict] = None
    risk_metrics: Optional[Dict] = None
    recommendation: Optional[str] = None


if __name__ == "__main__":
    # Test the model classes
    print("\nTesting Pydantic Models")
    print("----------------------")
    
    try:
        # 1. Test TickerInput
        print("\n1. Testing TickerInput model")
        ticker_input = TickerInput(ticker="AAPL")
        print(f"Created ticker input: {ticker_input.dict()}")
        
        # Test invalid ticker input
        try:
            invalid_ticker = TickerInput()
            print("Should not reach here - ticker is required")
        except Exception as e:
            print(f"Correctly caught missing ticker error: {str(e)}")
            
        # 2. Test DataInput
        print("\n2. Testing DataInput model")
        sample_data = {
            "data": {
                "historical_data": {
                    "Close": [100, 101, 102],
                    "Volume": [1000, 1100, 1200]
                },
                "company_info": {
                    "name": "Apple Inc.",
                    "sector": "Technology",
                    "market_cap": 2000000000000
                }
            }
        }
        
        data_input = DataInput(**sample_data)
        print("Created data input successfully")
        print("Sample data structure:")
        for key, value in data_input.data.items():
            print(f"\n{key}:")
            print(value)
            
        # 3. Test TradingState
        print("\n3. Testing TradingState model")
        trading_state = TradingState(
            ticker="AAPL",
            stock_data={
                "historical_data": {"Close": [100, 101, 102]},
                "company_info": {"name": "Apple Inc."}
            },
            technical_analysis={
                "sma_20": 100.5,
                "rsi": 65.4
            },
            prediction={
                "direction": "up",
                "probability": 0.75
            }
        )
        
        print("Created trading state with data:")
        print(f"Ticker: {trading_state.ticker}")
        print(f"Has stock data: {trading_state.stock_data is not None}")
        print(f"Has technical analysis: {trading_state.technical_analysis is not None}")
        print(f"Has prediction: {trading_state.prediction is not None}")
        
        # 4. Test model validation
        print("\n 4. Testing model validation")
        try:
            invalid_state = TradingState()  # Should fail - ticker is required
            print("Should not reach here - ticker is required")
        except Exception as e:
            print(f"Correctly caught missing ticker error: {str(e)}")
        
        print("\n All model tests completed successfully")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()