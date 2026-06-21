import pandas as pd

def compare_models(
    iforest_df: pd.DataFrame,
    svm_df: pd.DataFrame,
    ae_df: pd.DataFrame
):

    iforest_dates = set(
        iforest_df[
            iforest_df["isolationforest_anomaly"]
        ].index
    )

    svm_dates = set(
        svm_df[
            svm_df["oneclasssvm_anomaly"]
        ].index
    )

    ae_dates = set(
        ae_df[
            ae_df["autoencoder_anomaly"]
        ].index
    )

    # ==================================================
    # Counts
    # ==================================================

    iforest_count = len(iforest_dates)

    svm_count = len(svm_dates)

    ae_count = len(ae_dates)

    # ==================================================
    # Pairwise Overlaps
    # ==================================================

    iforest_svm = (
        iforest_dates &
        svm_dates
    )

    iforest_ae = (
        iforest_dates &
        ae_dates
    )

    svm_ae = (
        svm_dates &
        ae_dates
    )

    # ==================================================
    # Triple Overlap
    # ==================================================

    all_three = (
        iforest_dates &
        svm_dates &
        ae_dates
    )

    # ==================================================
    # Report
    # ==================================================

    print("\n" + "=" * 50)

    print("MODEL COMPARISON")

    print("=" * 50)

    print(
        f"Isolation Forest : "
        f"{iforest_count}"
    )

    print(
        f"One-Class SVM    : "
        f"{svm_count}"
    )

    print(
        f"Autoencoder      : "
        f"{ae_count}"
    )

    print("\nOverlaps")

    print(
        f"IForest ∩ SVM    : "
        f"{len(iforest_svm)}"
    )

    print(
        f"IForest ∩ AE     : "
        f"{len(iforest_ae)}"
    )

    print(
        f"SVM ∩ AE         : "
        f"{len(svm_ae)}"
    )

    print(
        f"All Three        : "
        f"{len(all_three)}"
    )

    if all_three:

        print(
            "\nConsensus Dates:"
        )

        for date in sorted(all_three):
            print(date)

    return {
        "iforest_count":
            iforest_count,

        "svm_count":
            svm_count,

        "ae_count":
            ae_count,

        "iforest_svm_overlap":
            sorted(iforest_svm),

        "iforest_ae_overlap":
            sorted(iforest_ae),

        "svm_ae_overlap":
            sorted(svm_ae),

        "all_three_overlap":
            sorted(all_three)
    }
