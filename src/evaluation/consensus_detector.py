import pandas as pd


def consensus_anomalies(
    iforest_df: pd.DataFrame,
    ae_df: pd.DataFrame,
    save_path=None
):
    """
    High-confidence anomalies detected by both
    Isolation Forest and Autoencoder.

    Returns
    -------
    pd.DataFrame
        DataFrame containing consensus anomalies
        and their associated scores.
    """

    iforest_dates = set(
        iforest_df[
            iforest_df["isolationforest_anomaly"]
        ].index
    )

    ae_dates = set(
        ae_df[
            ae_df["autoencoder_anomaly"]
        ].index
    )

    consensus_dates = sorted(
        iforest_dates & ae_dates
    )

    if not consensus_dates:

        print("\nNo consensus anomalies found.")

        return pd.DataFrame()

    consensus_df = pd.DataFrame(
        index=consensus_dates
    )

    consensus_df["Close"] = (
        iforest_df.loc[
            consensus_dates,
            "Close"
        ]
    )

    consensus_df["isolationforest_score"] = (
        iforest_df.loc[
            consensus_dates,
            "isolationforest_score"
        ]
    )

    consensus_df["autoencoder_score"] = (
        ae_df.loc[
            consensus_dates,
            "autoencoder_score"
        ]
    )

    print("\n" + "=" * 60)
    print("CONSENSUS ANOMALIES")
    print("=" * 60)

    print(
        f"High-confidence anomalies: "
        f"{len(consensus_df)}"
    )

    print("\nDates:")

    for date in consensus_df.index:
        print(date)

    if save_path is not None:

        consensus_df.to_csv(
            save_path
        )

        print(
            f"\nSaved to: {save_path}"
        )

    return consensus_df
