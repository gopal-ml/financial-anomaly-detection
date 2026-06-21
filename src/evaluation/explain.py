def explain_anomaly(row):

    reasons = []

    if abs(row["return_zscore"]) > 2:
        reasons.append(
            f"Return z-score={row['return_zscore']:.2f}"
        )

    if row["volume_ratio"] > 2:
        reasons.append(
            f"Volume {row['volume_ratio']:.2f}x average"
        )

    if row["price_ma50_ratio"] < 0.9:
        reasons.append(
            "Price significantly below trend"
        )

    return reasons
