from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def plot_detected_anomalies(
    df: pd.DataFrame,
    ticker: str,
    anomaly_col: str,
    price_col: str = "Close",
    model : str = "Isolation Forest",
    save_path = None,
    show : bool = True
):
    
    anomalies = df[
        df[anomaly_col]
    ]
    
    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        df.index,
        df[price_col],
        color="steelblue",
        linewidth=2,
        label="Close Price"
    )


    ax.scatter(
        anomalies.index,
        anomalies[price_col],
        color="crimson",
        edgecolors="black",
        s=100,
        label="Detected Anomalies",
        zorder=3
    )

    ax.grid(
        alpha=0.3,
        linestyle="--"
    )

    ax.legend(
        loc="best",
        frameon=True
    )

    anomaly_count = len(anomalies)

    ax.set_title(
        f"{ticker} - {model} "
        f"({anomaly_count} anomalies)"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.legend()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.tight_layout()

    if show:
        plt.show()
    
    plt.close()

def plot_top_anomalies(
    df: pd.DataFrame,
    ticker: str,
    price_col: str = "Close",
    score_col : str = "iforest_score",
    model : str = "Isolation Forest",
    ascending : bool = True,
    top_n : int = 20,
    show: bool = True
):
    
    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        df.index,
        df[price_col],
        color="steelblue",
        linewidth=2,
        label="Close Price"
    )

    top_anomalies = (
        df
        .sort_values(score_col, ascending= ascending)
        .head(top_n)
    )


    ax.scatter(
        top_anomalies.index,
        top_anomalies[price_col],
        color="crimson",
        edgecolors="black",
        s=100,
        label= f"Top {top_n} scores",
        zorder=3
    )

    ax.set_title(
        f"{ticker} — Top {top_n} {model} Scores"
    )
    
    ax.grid(
        alpha=0.3,
        linestyle="--"
    )

    ax.legend(
        loc="best",
        frameon=True
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    ax.legend()

    plt.tight_layout()

    if show:
        plt.show()
    
    plt.close()
