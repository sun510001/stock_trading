import pandas as pd
import numpy as np
import os
from typing import List, Dict, Any, Optional
from logger import logger
from data_loader.yahoo_downloader import YahooIncrementalLoader
from utils.naming import sanitize_filename

class DataProcessor:
    """
    [ETL Module]
    Responsible for processing raw CSV data into a clean, synchronized portfolio matrix.
    
    This class includes financial engineering engines to convert raw market yields into 
    synthetic total return price series and handles temporal alignment across multiple assets.
    """

    def __init__(self, raw_path: str = "./data", processed_path: str = "./data_processed") -> None:
        """
        Initialize the DataProcessor with paths for raw and processed data.

        Args:
            raw_path (str): Directory containing raw asset CSV files. Defaults to "./data".
            processed_path (str): Directory where processed artifacts will be saved. 
                                 Defaults to "./data_processed".
        """
        self.raw_path: str = raw_path
        self.processed_path: str = processed_path
        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)

    def _load_raw(self, safe_name: str) -> pd.Series:
        """
        Load the 'Close' price column from a raw asset CSV file.

        Args:
            safe_name (str): The sanitized filename (without extension).

        Returns:
            pd.Series: A time series of closing prices indexed by Date.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
        """
        path = os.path.join(self.raw_path, f"{safe_name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Raw data file not found: {path}")
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df = df[~df.index.duplicated(keep="last")]
        df.sort_index(inplace=True)
        if "Close" in df.columns:
            return df["Close"]
        return df.iloc[:, 0]

    def bond_pricing_engine(self, yield_series: pd.Series, duration: float = 20.0, initial_price: float = 100.0) -> pd.Series:
        y = yield_series / 100.0
        dy = y.diff().fillna(0)
        interest_income = y.shift(1).fillna(y.iloc[0]) / 252.0
        capital_gain = -duration * dy
        total_daily_return = interest_income + capital_gain
        price_series = initial_price * (1 + total_daily_return).cumprod()
        return price_series

    def cash_pricing_engine(self, yield_series: pd.Series, initial_price: float = 100.0) -> pd.Series:
        y = yield_series / 100.0
        daily_ret = y.shift(1).fillna(y.iloc[0]) / 252.0
        price_series = initial_price * (1 + daily_ret).cumprod()
        return price_series

    def build_aligned_dataframe(self, assets: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build aligned price matrix for given assets and return as DataFrame.

        全量对齐矩阵中允许存在 NaN；只去掉整行全空的日期。具体的“木桶式裁剪”
        会在回测阶段按本次使用的资产子集进行。"""
        logger.info("Starting Multi-Asset Data Processing & Alignment (in-memory)...")
        prices: Dict[str, pd.Series] = {}

        for asset in assets:
            name = asset["name"]
            kind = asset.get("kind", "price")
            engine = asset.get("engine")

            duration: float = 20.0
            if kind == "yield" and engine == "bond":
                raw_duration = asset.get("duration", 20.0)
                if raw_duration is not None:
                    duration = float(raw_duration)

            safe_name = sanitize_filename(name)
            logger.info(f"Processing asset '{name}' (source: {safe_name}.csv)...")

            raw_series = self._load_raw(safe_name)

            if kind == "price":
                price_series = raw_series
            elif kind == "yield":
                if engine == "bond":
                    price_series = self.bond_pricing_engine(raw_series, duration=duration)
                elif engine == "cash":
                    price_series = self.cash_pricing_engine(raw_series)
                else:
                    raise ValueError(f"Unknown engine '{engine}' for asset '{name}'")
            else:
                raise ValueError(f"Unsupported asset kind '{kind}' for asset '{name}'")

            prices[name] = price_series

        if not prices:
            raise ValueError("No assets were successfully processed.")

        portfolio_df = pd.DataFrame(prices)
        original_len = len(portfolio_df)
        # 只丢弃整行全为空的日期，保留部分资产缺失的数据，
        # 以便在回测阶段按具体资产子集再做裁剪。
        portfolio_df.dropna(how="all", inplace=True)
        final_len = len(portfolio_df)

        logger.info(f"Alignment complete. Rows: {original_len} -> {final_len}")
        logger.info(
            f"Available range: {portfolio_df.index.min().date()} to {portfolio_df.index.max().date()}"
        )
        return portfolio_df

    def process_and_align(self, assets: List[Dict[str, Any]], output_filename: str = "aligned_assets.csv") -> str:
        """Full ETL pipeline: build aligned DataFrame and persist to CSV.

        Args:
            assets: Asset configuration list.
            output_filename: Target CSV filename under processed_path.

        Returns:
            str: Full path to the written CSV file.
        """
        try:
            portfolio_df = self.build_aligned_dataframe(assets)
            output_file = os.path.join(self.processed_path, output_filename)
            portfolio_df.to_csv(output_file)
            logger.info(f"Aligned assets saved successfully to: {output_file}")
            return output_file
        except Exception as e:
            logger.exception(f"Data pipeline processing failed: {str(e)}")
            raise


if __name__ == "__main__":
    sample_assets = [
        {"name": "Stocks", "ticker": "^NDX", "kind": "price"},
        {"name": "Gold", "ticker": "^XAU", "kind": "price"},
        {"name": "Bonds", "ticker": "^TYX", "kind": "yield", "engine": "bond", "duration": 20.0},
        {"name": "Cash", "ticker": "^IRX", "kind": "yield", "engine": "cash"},
    ]
    processor = DataProcessor()
    processor.process_and_align(sample_assets)
