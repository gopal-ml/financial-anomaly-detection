import yfinance as yf
from pathlib import Path
from configs.settings import (
    DATA_DIR,
    MODEL_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

def download_stock_data(
    ticker: str,
    start: str,
    end: str,
    save_dir: str = RAW_DATA_DIR
):

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    file_path = save_path / f"{ticker}.csv"

    df.to_csv(file_path)

    print(f"Saved {ticker} to {file_path}")

    return df


if __name__ == "__main__":

    download_stock_data(
        ticker="NVDA",
        start="2015-01-01",
        end="2025-01-01"
    )

