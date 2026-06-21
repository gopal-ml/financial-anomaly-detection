import pandas as pd
import numpy as np

def load_data(path: str):

    df = pd.read_csv(path)

    df = df.iloc[2:].reset_index(drop=True)

    numeric_cols = [
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df.dropna(
        subset=numeric_cols,
        inplace=True
    )

    df["Price"] = pd.to_datetime(
        df["Price"]
    )

    df.rename(
        columns={"Price": "Date"},
        inplace=True
    )

    df.sort_values(
        "Date",
        inplace=True
    )

    df.drop_duplicates(
        subset="Date",
        keep="last",
        inplace=True
    )

    df.set_index(
        "Date",
        inplace=True
    )
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def create_features(df: pd.DataFrame):

    df = df.copy()

    # returns
    df["return"] = df["Close"].pct_change()

    # log returns
    df["log_return"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # volume features
    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    df["volume_ma20"] = (
        df["Volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["Volume"]
        / df["volume_ma20"]
    )

    # volatility
    df["volatility_7"] = (
        df["return"]
        .rolling(7)
        .std()
    )

    df["volatility_30"] = (
        df["return"]
        .rolling(30)
        .std()
    )

    # moving averages
    df["ma10"] = (
        df["Close"]
        .rolling(10)
        .mean()
    )

    df["ma50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["price_ma10_ratio"] = (
        df["Close"]
        / df["ma10"]
    )

    df["price_ma50_ratio"] = (
        df["Close"]
        / df["ma50"]
    )

    # z-score
    rolling_mean = (
        df["return"]
        .rolling(30)
        .mean()
    )

    rolling_std = (
        df["return"]
        .rolling(30)
        .std()
    )

    df["return_zscore"] = (
        (df["return"] - rolling_mean)
        / rolling_std
    )

    df.dropna(inplace=True)

    return df

from pathlib import Path

def save_processed_data(
    df,
    ticker,
    save_dir="data/processed"
):

    Path(save_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        f"{save_dir}/{ticker}_features.csv"
    )
