from data_loader.yahoo_downloader import YahooIncrementalLoader
from data_loader.akshare_downloader import AkshareIncrementalLoader
from data_loader.data_processor import DataProcessor
from backend.assets_config import AssetConfigManager
from utils.naming import sanitize_filename
from logger import logger
import os
import pandas as pd


class DataPipelineRunner:
    """
    Runner class to orchestrate the end-to-end data pipeline.

    This includes loading asset configurations, downloading raw data from
    external sources, and processing/aligning the data for backtesting.
    """

    def __init__(self, start_year: int = 1985) -> None:
        """
        Initialize the pipeline runner.

        Args:
            start_year (int): The default start year for data download if no
                              specific date is configured. Defaults to 1985.
        """
        self.start_year: int = start_year
        self.storage_path: str = "./data"

    def _load_local_data(self, name: str) -> pd.DataFrame:
        safe_name = sanitize_filename(name)
        file_path = os.path.join(self.storage_path, f"{safe_name}.csv")
        if not os.path.exists(file_path):
            return pd.DataFrame()
        try:
            return pd.read_csv(file_path, index_col="Date", parse_dates=True)
        except Exception:
            return pd.DataFrame()

    def run(self) -> None:
        """
        Execute the data pipeline: Load -> Download -> Process.

        This method coordinates the sequence of operations required to prepare
        market data for the backtest engine.
        """
        assets = AssetConfigManager.load_assets()
        if not assets:
            logger.error(
                "No assets configured. Please add assets via the UI or config/assets.json first."
            )
            return

        logger.info(">>> STEP 1: DOWNLOADING RAW DATA (INCREMENTAL) <<<")
        yahoo_downloader = YahooIncrementalLoader(storage_path=self.storage_path)
        akshare_downloader = AkshareIncrementalLoader(storage_path=self.storage_path)

        import datetime as _dt

        for asset in assets:
            name = asset["name"]
            ticker = asset["ticker"]
            asset_start_date = asset.get("initial_start_date") or asset.get("start_date")
            start_fallback = asset_start_date if asset_start_date else f"{self.start_year}-01-01"

            logger.info(f"Processing asset: {name} ({ticker})")

            yahoo_downloader.download_symbol(ticker, name, start_fallback)
            df_yahoo = self._load_local_data(name)

            if df_yahoo.empty:
                logger.info(
                    f"[{name}] No Yahoo data available after download, falling back to AkShare."
                )
                akshare_downloader.download_symbol(ticker, name, start_fallback)
                continue

            start_required = _dt.datetime.strptime(start_fallback, "%Y-%m-%d").date()
            yahoo_start = df_yahoo.index[0].date()
            yahoo_end = df_yahoo.index[-1].date()

            yahoo_span = (yahoo_end - yahoo_start).days

            today = _dt.date.today()
            ak_start_str = start_required.strftime("%Y%m%d")
            ak_end_str = today.strftime("%Y%m%d")

            df_ak = akshare_downloader._us_loader._fetch_single_symbol(
                ticker, ak_start_str, ak_end_str
            )

            if df_ak.empty:
                logger.info(f"[{name}] AkShare returned no data, keep Yahoo data.")
                continue

            ak_start = df_ak.index[0].date()
            ak_end = df_ak.index[-1].date()
            ak_span = (ak_end - ak_start).days

            logger.info(
                f"[{name}] Yahoo span: {yahoo_start} -> {yahoo_end} ({yahoo_span} days); "
                f"AkShare span: {ak_start} -> {ak_end} ({ak_span} days)."
            )

            if ak_span > yahoo_span:
                logger.info(f"[{name}] AkShare coverage longer, replacing local data with AkShare.")
                df_ak.index.name = "Date"
                file_path = os.path.join(self.storage_path, f"{sanitize_filename(name)}.csv")
                df_ak.to_csv(file_path)
            else:
                logger.info(f"[{name}] Yahoo coverage longer or equal, keeping Yahoo data.")

        logger.info(">>> STEP 2: PROCESSING & SYNTHETIC PRICING <<<")
        processor = DataProcessor(raw_path=self.storage_path, processed_path="./data_processed")
        processor.process_and_align(assets)

        logger.info(">>> DATA PIPELINE FINISHED <<<")


if __name__ == "__main__":
    runner = DataPipelineRunner(start_year=1985)
    runner.run()
