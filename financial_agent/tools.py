from typing import Dict
import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime
from langchain_core.tools import tool
from models import TickerInput, DataInput
from data import MarketData
from sklearn.ensemble import RandomForestClassifier


class TradingTools:
    """Trading analysis tools"""

    def __init__(self, api_key: str):
        self.market_data_fetcher = MarketData(api_key)

    def fetch_market_data(self, ticker: str) -> Dict:
        """Fetch current market data for analysis"""
        try:
            data = self.market_data_fetcher.fetch_stock_data(ticker)
            return {
                "historical_data": data.to_dict(),
                "company_info": {"name": ticker}  # placeholder if no info
            }
        except Exception as e:
            print(f"Error fetching market data: {e}")
            raise

    @tool(args_schema=DataInput)
    def analyze_technicals(self, data: Dict) -> Dict:
        """Perform technical analysis on market data"""
        try:
            df = pd.DataFrame(data["historical_data"])
            df['Close'] = pd.to_numeric(df['close'], errors='coerce')
            df.dropna(inplace=True)

            return {
                "sma_20": float(ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]),
                "sma_50": float(ta.trend.sma_indicator(df['Close'], window=50).iloc[-1]),
                "rsi": float(ta.momentum.rsi(df['Close']).iloc[-1]),
                "macd": float(ta.trend.macd_diff(df['Close']).iloc[-1])
            }
        except Exception as e:
            print(f"Error in technical analysis: {e}")
            raise

    @tool(args_schema=DataInput)
    def predict_movement(self, data: Dict) -> Dict:
        """Predict price movement probability using ML"""
        df = pd.DataFrame(data["historical_data"])
        df['Close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(inplace=True)

        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['sma20_ratio'] = df['Close'] / df['Close'].rolling(20).mean()
        df['rsi'] = ta.momentum.rsi(df['Close'])
        df['macd_diff'] = ta.trend.macd_diff(df['Close'])
        df['bb_width'] = (ta.volatility.bollinger_hband(df['Close']) -
                          ta.volatility.bollinger_lband(df['Close'])) / df['Close']

        features = ['volatility', 'sma20_ratio', 'rsi', 'macd_diff', 'bb_width']
        df.dropna(inplace=True)
        X = df[features]
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        y = df['target'].dropna()
        X = X.iloc[:-1]

        if len(X) < 50:
            raise ValueError("Not enough data for prediction")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        train_size = int(len(X) * 0.8)
        model.fit(X[:train_size], y[:train_size])
        latest_features = X.iloc[-1:].values
        proba = model.predict_proba(latest_features)[0]

        recent_momentum = df['returns'].tail(5).mean()
        volatility = df['returns'].std() * np.sqrt(252)

        return {
            "upward_probability": float(proba[1]),
            "downward_probability": float(proba[0]),
            "confidence_score": float(max(proba[0], proba[1])),
            "recent_momentum": float(recent_momentum),
            "volatility": float(volatility),
            "prediction": "up" if proba[1] > proba[0] else "down",
            "feature_importance": dict(zip(features, model.feature_importances_))
        }

    @tool(args_schema=TickerInput)
    def analyze_options(self, ticker: str) -> Dict:
        """Analyze options trading opportunities"""
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice')
        if not current_price:
            return {"error": "Unable to fetch current price"}

        options_analysis = {}
        try:
            if not stock.options:
                return {"error": "No options data available"}

            for expiration in stock.options[:3]:
                chain = stock.option_chain(expiration)
                calls = chain.calls
                puts = chain.puts

                atm_call = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:1]]
                atm_put = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:1]]

                iv_rank = (atm_call['impliedVolatility'].iloc[0] - calls['impliedVolatility'].min()) / \
                          (calls['impliedVolatility'].max() - calls['impliedVolatility'].min())

                options_analysis[expiration] = {
                    "current_price": float(current_price),
                    "days_to_expiry": (datetime.strptime(expiration, '%Y-%m-%d') - datetime.now()).days,
                    "calls": {
                        "strike": float(atm_call['strike'].iloc[0]),
                        "price": float(atm_call['lastPrice'].iloc[0]),
                        "implied_volatility": float(atm_call['impliedVolatility'].iloc[0]),
                        "volume": int(atm_call['volume'].iloc[0]) if not pd.isna(atm_call['volume'].iloc[0]) else 0,
                        "open_interest": int(atm_call['openInterest'].iloc[0]) if not pd.isna(atm_call['openInterest'].iloc[0]) else 0
                    },
                    "puts": {
                        "strike": float(atm_put['strike'].iloc[0]),
                        "price": float(atm_put['lastPrice'].iloc[0]),
                        "implied_volatility": float(atm_put['impliedVolatility'].iloc[0]),
                        "volume": int(atm_put['volume'].iloc[0]) if not pd.isna(atm_put['volume'].iloc[0]) else 0,
                        "open_interest": int(atm_put['openInterest'].iloc[0]) if not pd.isna(atm_put['openInterest'].iloc[0]) else 0
                    },
                    "market_sentiment": {
                        "put_call_ratio": float(puts['volume'].sum() / calls['volume'].sum()) if calls['volume'].sum() > 0 else 0,
                        "iv_rank": float(iv_rank),
                        "iv_percentile": float(np.percentile(calls['impliedVolatility'], 50))
                    }
                }

            return options_analysis

        except Exception as e:
            return {"error": f"Unable to fetch options data: {str(e)}"}

    @tool(args_schema=DataInput)
    def calculate_risk_metrics(self, data: Dict) -> Dict:
        """Calculate comprehensive risk metrics"""
        df = pd.DataFrame(data["historical_data"])
        df['Close'] = pd.to_numeric(df['close'], errors='coerce')
        df.dropna(inplace=True)
        returns = df['Close'].pct_change().dropna()

        volatility = returns.std() * np.sqrt(252)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean()
        cvar_99 = returns[returns <= var_99].mean()

        rolling_max = df['Close'].expanding().max()
        drawdown = (df['Close'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        risk_free_rate = 0.03
        excess_returns = returns - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0

        downside_returns = returns[returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() if len(downside_returns) > 0 else 0

        account_size = 100000
        risk_per_trade = 0.02

        return {
            "volatility_metrics": {
                "daily_volatility": float(returns.std()),
                "annual_volatility": float(volatility),
                "up_volatility": float(returns[returns > 0].std()),
                "down_volatility": float(returns[returns < 0].std())
            },
            "value_at_risk": {
                "var_95": float(var_95),
                "var_99": float(var_99),
                "cvar_95": float(cvar_95),
                "cvar_99": float(cvar_99)
            },
            "drawdown_metrics": {
                "max_drawdown": float(max_drawdown),
                "current_drawdown": float(drawdown.iloc[-1]),
                "avg_drawdown": float(drawdown.mean())
            },
            "performance_metrics": {
                "sharpe_ratio": float(sharpe_ratio),
                "sortino_ratio": float(sortino_ratio),
                "calmar_ratio": float(returns.mean() * 252 / abs(max_drawdown)) if max_drawdown != 0 else 0
            },
            "position_sizing": {
                "account_size": account_size,
                "max_position_size": float(account_size * risk_per_trade / abs(var_95)) if var_95 != 0 else 0,
                "suggested_stop_loss": float(df['Close'].iloc[-1] * (1 - risk_per_trade)),
                "max_loss_allowed": float(account_size * risk_per_trade)
            }
        }