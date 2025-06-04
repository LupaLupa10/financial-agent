import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Tuple, Any, List
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
from statsmodels.tsa.stattools import coint

from utils.utils import ensure_directory_exists
load_dotenv()


class MarketData:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key not found.")
        self.ts = TimeSeries(key=api_key, output_format='pandas')

    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        # data, _ = self.ts.get_daily(symbol=ticker, outputsize='full')
        data, _ = self.ts.get_daily_adjusted(symbol=ticker, outputsize='full')
        data.columns = ['open', 'high', 'low', 'close', 'volume']
        data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        return data

    def calculate_spread_and_zscore(self, series1: pd.Series, series2: pd.Series, window: int = 20) -> pd.DataFrame:
        slope, intercept = np.polyfit(series1, series2, 1)
        spread = series1 - slope * series2 + intercept
        zscore = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
        return pd.DataFrame({"spread": spread, "zscore": zscore})
        
    def is_cointegrated(self, series1: pd.Series, series2: pd.Series, threshold: float = 0.05) -> bool:
        score, pvalue, _ = coint(series1, series2)
        return pvalue < threshold

    def calculate_cointegrated_pairs(self, stock_data: Dict[str, pd.DataFrame], threshold: float = 0.05) -> List[Tuple[str, str]]:
        tickers = list(stock_data.keys())
        cointegrated_pairs = []

        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                s1 = stock_data[tickers[i]]['close']
                s2 = stock_data[tickers[j]]['close']

                if self.is_cointegrated(s1, s2, threshold):
                    cointegrated_pairs.append((tickers[i], tickers[j]))

        return cointegrated_pairs


class MarketVisualization:
    @staticmethod
    def plot_dual_price(df1: pd.Series, df2: pd.Series, label1='Stock1', label2='Stock2'):
        plt.figure(figsize=(10, 5))
        plt.plot(df1.index, df1, label=label1)
        plt.plot(df2.index, df2, label=label2)
        plt.title(f"{label1} vs {label2} Prices")
        plt.legend()
        plt.savefig(os.path.join(IMG_DIR, f"{label1}_vs_{label2}_prices.png"))
        plt.show()
        plt.close()

    @staticmethod
    def plot_spread(df: pd.DataFrame, upper=2, lower=-2):
        """
        Spread is the difference between two stocks. It tracks the difference between the two stocks over time.
        The mean is the average difference between the two stocks, also the equilibrium point.
        The upper and lower bounds are the points where the spread is 2 standard deviations from the mean.
        If the spread is above the upper bound, it means that the stock is overvalued relative to the other stock. 
        If the spread is below the lower bound, it means that the stock is undervalued relative to the other stock.
        Shorting the overvalued stock and buying the undervalued stock is a good strategy to profit from the spread.
        """
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['spread'], label='Spread')
        plt.axhline(df['spread'].mean(), color='black', linestyle='--', label='Mean')
        plt.axhline(df['spread'].mean() + upper * df['spread'].std(), color='red', linestyle='--', label='+2σ')
        plt.axhline(df['spread'].mean() - lower * df['spread'].std(), color='green', linestyle='--', label='-2σ')
        plt.title('Price Spread with Entry/Exit Thresholds')
        plt.legend()
        plt.savefig(os.path.join(IMG_DIR, "spread.png"))
        plt.show()
        plt.close()

    @staticmethod
    def plot_zscore_with_signals(df: pd.DataFrame, label1='Stock1', label2='Stock2'):
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df['zscore'], label='Z-score')
        plt.axhline(0, color='black')
        plt.axhline(2, color='red', linestyle='--')
        plt.axhline(-2, color='green', linestyle='--')

        # Plot signals
        buy_signals = df[df['zscore'] < -2]
        sell_signals = df[df['zscore'] > 2]

        plt.scatter(buy_signals.index, buy_signals['zscore'], color='green', label='Long Signal', marker='^')
        plt.scatter(sell_signals.index, sell_signals['zscore'], color='red', label='Short Signal', marker='v')

        plt.title("Z-score and Trading Signals")
        plt.legend()
        plt.savefig(os.path.join(IMG_DIR, f"{label1}_vs_{label2}_zscores.png"))
        plt.show()
        plt.close()


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    IMG_DIR = os.path.join(BASE_DIR, "..", "images")
    ensure_directory_exists(DATA_DIR, IMG_DIR)

    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY")
    market_data = MarketData(api_key)

    # Example: Fetch multiple stocks
    tickers = ["KO", "PEP"]
    stock_data = {}
    for ticker in tickers:
        try:
            stock_data[ticker] = market_data.fetch_stock_data(ticker)
        except Exception as e:
            print(f"Failed to fetch {ticker}: {e}")

    stock_data_df = pd.DataFrame([stock_data])
    stock_data_df.to_csv(os.path.join(DATA_DIR, "stock_data.csv"))
    
    for ticker, df in stock_data.items():
        df.to_csv(os.path.join(DATA_DIR, f"{ticker}.csv"))
        
    stock_data = {
        ticker: pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), index_col=0, parse_dates=True)
        for ticker in tickers
    }
    
    # Find cointegrated pairs
    pairs = market_data.calculate_cointegrated_pairs(stock_data)
    print("\nCointegrated Pairs Found:")
    for p in pairs:
        print(p)

    if pairs:
        s1, s2 = pairs[0]
        close1 = stock_data[s1]['close']
        close2 = stock_data[s2]['close']
        df = market_data.calculate_spread_and_zscore(close1, close2)

        MarketVisualization.plot_dual_price(close1, close2, label1=s1, label2=s2)
        MarketVisualization.plot_spread(df)
        MarketVisualization.plot_zscore_with_signals(df, label1=s1, label2=s2)

    # Optional: Plot z-score for a pair
    # if pairs:
    #     s1, s2 = pairs[0]
    #     df = market_data.calculate_spread_and_zscore(stock_data[s1]['close'], stock_data[s2]['close'])
    #     df[['zscore']].plot(title=f"Z-score of Spread: {s1} vs {s2}", figsize=(10, 5))
    #     plt.axhline(2, color='red', linestyle='--')
    #     plt.axhline(-2, color='green', linestyle='--')
    #     plt.axhline(0, color='black')
    #     plt.show()
