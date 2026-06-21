def train_test_split_time_series(
    df,
    train_ratio=0.8
):
    split_idx = int(
        len(df) * train_ratio
    )

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df
