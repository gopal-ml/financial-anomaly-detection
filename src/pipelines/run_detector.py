from configs.settings import (
    MODEL_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODEL_NAMES,
    REPORT_DIR
)

from src.data.preprocess import (
    load_data,
    create_features,
    save_processed_data
)
from configs.features import FEATURES
from src.data import scaler, split
from src.visualization.plot import plot_detected_anomalies
from src.visualization.training_loss import plot_training_loss
from src.models.base_detector import BaseDetector



def run_detector(
    detector: BaseDetector,
    ticker: str
):

    model_name = (
        detector.__class__.__name__
        .replace("Detector", "")
        .lower()
    )

    display_name = MODEL_NAMES.get(
        model_name,
        model_name
    )

    print(
        f"\nRunning {display_name} on {ticker}"
    )

    # =====================================================
    # Load & Feature Engineering
    # =====================================================

    df = load_data(
        RAW_DATA_DIR/f"{ticker}.csv"
    )

    features_df = create_features(df)

    save_processed_data(
        features_df,
        ticker=ticker,
        save_dir=PROCESSED_DATA_DIR
    )

    # =====================================================
    # Split & Scale
    # =====================================================

    train_df, test_df = split.train_test_split_time_series(features_df)

    train_X = train_df[FEATURES]
    test_X = test_df[FEATURES]
    
    train_X_scaled, test_X_scaled,  fitted_scaler= scaler.scale_data(train_X, test_X)

    # =====================================================
    # Train
    # =====================================================

    detector.fit(train_X_scaled)

    # Autoencoder-specific visualization

    if model_name == "autoencoder":
        plot_training_loss(
            detector.loss_history,
            ticker=ticker,
            save_path=(
                REPORT_DIR / f"figures/"
                f"{ticker}_ae_loss.png"
            ),
            show = False
        )

    # =====================================================
    # Predict
    # =====================================================

    test_preds = detector.predict(test_X_scaled)

    test_scores = detector.anomaly_score(test_X_scaled)

    anomaly_col = (f"{model_name}_anomaly")

    score_col = (f"{model_name}_score")

    test_df[anomaly_col] = test_preds

    test_df[score_col] = test_scores

    # =====================================================
    # Save Results
    # =====================================================

    test_df.to_csv(
        PROCESSED_DATA_DIR /
        f"{ticker}_{model_name}_results.csv"
    )

    # =====================================================
    # Save Model
    # =====================================================
    try:
        extension = (
            "pth"
            if model_name == "autoencoder"
            else "pkl"
        )

        detector.save(
            MODEL_DIR /
            f"{ticker}_{model_name}.{extension}"
        )

    except Exception as e:
        print(
            f"Model save failed: {e}"
        )

    # =====================================================
    # Visualization
    # =====================================================
    
    if model_name == "autoencoder":
        ascending = False
    else:
        ascending = True

    plot_detected_anomalies(
        df=test_df,
        ticker= ticker,
        anomaly_col= anomaly_col,
        model= display_name,
        save_path=(
                REPORT_DIR / f"figures/"
                f"{ticker}_{model_name}_anomalies.png"
            ),
        show= False
    )

    print(
        f"Anomalies detected: "
        f"{test_df[anomaly_col].sum()}"
    )

    return test_df

if __name__ == "__main__":

    from src.models.auto_encoder import AutoEncoderDetector

    run_detector(
        detector= AutoEncoderDetector(
            input_dim=8,
            device="cpu"
            ),
        ticker= "AAPL"
    )
