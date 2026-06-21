from src.models.isolation_forest import (
    IsolationForestDetector
)

from src.models.one_class_svm import (
    OneClassSVMDetector
)

from src.models.auto_encoder import (
    AutoEncoderDetector
)

from src.pipelines.run_detector import (
    run_detector
)

from src.evaluation.compare_models import (
    compare_models
)

from src.evaluation.model_report import (
    generate_model_report
)

from src.evaluation.consensus_detector import (
    consensus_anomalies
)

from configs.features import (
    FEATURES
)

from configs.settings import (
    REPORT_DIR
)

def run_all_models(ticker: str = "AAPL"):
    print(
        f"\n{'=' * 60}"
    )
    print(
        f"RUNNING ALL MODELS FOR {ticker}"
    )
    print(
        f"{'=' * 60}\n"
    )

    iforest_df = run_detector(
        IsolationForestDetector(),
        ticker
    )

    svm_df = run_detector(
        OneClassSVMDetector(),
        ticker
    )

    ae_df = run_detector(
        AutoEncoderDetector(
            input_dim=len(FEATURES),
            device= 'cpu'
        ),
        ticker
    )
    # ==========================================
    # Compare Models
    # ==========================================
    comparison_results = compare_models(
        iforest_df,
        svm_df,
        ae_df
    )
    # ==========================================
    # Consensus Detection
    # ==========================================
    consensus_df = consensus_anomalies(
        iforest_df,
        ae_df,
        save_path=(
            REPORT_DIR /
            "results" /
            f"{ticker}_consensus.csv"
        )
    )
    # ==========================================
    # Generate Report
    # ==========================================
    generate_model_report(
        ticker=ticker,
        comparison_results=comparison_results
    )

    print(
        f"\nFinished processing {ticker}"
    )
    
    return {
        "iforest_df": iforest_df,
        "svm_df": svm_df,
        "ae_df": ae_df,
        "consensus_df": consensus_df,
        "comparison_results": comparison_results
    }


if __name__ == "__main__":

    run_all_models()
