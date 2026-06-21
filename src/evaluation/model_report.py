from configs.settings import REPORT_DIR

def generate_model_report(
    ticker: str,
    comparison_results: dict,
    save_report: bool = True
):

    report = []

    report.append("=" * 60)
    report.append("FINANCIAL ANOMALY DETECTION REPORT")
    report.append("=" * 60)

    report.append(f"Ticker: {ticker}")
    report.append("")

    report.append("MODEL RESULTS")
    report.append("-" * 60)

    report.append(
        f"Isolation Forest : "
        f"{comparison_results['iforest_count']}"
    )

    report.append(
        f"One-Class SVM    : "
        f"{comparison_results['svm_count']}"
    )

    report.append(
        f"Autoencoder      : "
        f"{comparison_results['ae_count']}"
    )

    report.append("")

    report.append("OVERLAPS")
    report.append("-" * 60)

    report.append(
        f"IForest ∩ SVM : "
        f"{len(comparison_results['iforest_svm_overlap'])}"
    )

    report.append(
        f"IForest ∩ AE  : "
        f"{len(comparison_results['iforest_ae_overlap'])}"
    )

    report.append(
        f"SVM ∩ AE      : "
        f"{len(comparison_results['svm_ae_overlap'])}"
    )

    report.append(
        f"All Three     : "
        f"{len(comparison_results['all_three_overlap'])}"
    )

    report.append("")

    report.append("CONSENSUS DATES")
    report.append("-" * 60)

    if comparison_results["all_three_overlap"]:

        for date in comparison_results[
            "all_three_overlap"
        ]:

            report.append(str(date))

    else:

        report.append(
            "No consensus anomalies found."
        )

    report_text = "\n".join(report)

    print("\n")
    print(report_text)

    if save_report:

        report_path = (
            REPORT_DIR /
            f"results/{ticker}_report.txt"
        )

        with open(
            report_path,
            "w"
        ) as f:

            f.write(report_text)

        print(
            f"\nReport saved to:\n{report_path}"
        )

    return report_text
