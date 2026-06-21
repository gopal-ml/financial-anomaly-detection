import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from explain import explain_anomaly
# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Financial Anomaly Detection",
    layout="wide"
)

st.title(
    "Financial Anomaly Detection Dashboard"
)

st.markdown(
"""
This dashboard visualizes anomalies detected
using Isolation Forest, One-Class SVM,
and Deep Autoencoders on stock market data.
"""
)

# ==================================================
# HELPERS
# ==================================================

@st.cache_data
def load_model_results(ticker):

    iforest_df = pd.read_csv(
        f"data/processed/{ticker}_isolationforest_results.csv"
    )

    svm_df = pd.read_csv(
        f"data/processed/{ticker}_oneclasssvm_results.csv"
    )

    ae_df = pd.read_csv(
        f"data/processed/{ticker}_autoencoder_results.csv"
    )

    consensus_df = pd.read_csv(
        f"reports/results/{ticker}_consensus.csv"
    )

    return (
        iforest_df,
        svm_df,
        ae_df,
        consensus_df
    )


def create_anomaly_chart(
    df,
    anomaly_col,
    score_col,
    title
):

    fig = go.Figure()

    # Price line

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price",
            line = dict(
                color = 'royalblue',
                width = 2
            )
        )
    )

    # Anomalies

    anomalies = df[
        df[anomaly_col]
    ]

    fig.add_trace(
        go.Scatter(
            x=anomalies["Date"],
            y=anomalies["Close"],
            mode="markers",
            name="Anomalies",
            marker=dict(
                color="red",
                size=8,
                line = dict(
                    color = "black",
                    width = 2
                )
            ),
            text=anomalies[score_col],
            hovertemplate=
            "<b>Date:</b> %{x}<br>"
            "<b>Price:</b> %{y:.2f}<br>"
            "<b>Score:</b> %{text:.6f}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        )
    )

    return fig


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header(
    "Configuration"
)

ticker = st.sidebar.selectbox(
    "Ticker",
    [
        "AAPL",
        "MSFT",
        "NVDA",
        "TSLA"
    ]
)

model = st.sidebar.selectbox(
    "Model",
    [
        "Isolation Forest",
        "One-Class SVM",
        "Autoencoder"
    ]
)


# ==================================================
# LOAD DATA
# ==================================================

(
    iforest_df,
    svm_df,
    ae_df,
    consensus_df
) = load_model_results(ticker)


# ==================================================
# COUNTS
# ==================================================

iforest_count = (
    iforest_df[
        iforest_df[
            "isolationforest_anomaly"
        ]
    ]
    .shape[0]
)

svm_count = (
    svm_df[
        svm_df[
            "oneclasssvm_anomaly"
        ]
    ]
    .shape[0]
)

ae_count = (
    ae_df[
        ae_df[
            "autoencoder_anomaly"
        ]
    ]
    .shape[0]
)

consensus_count = len(
    consensus_df
)


# ==================================================
# METRICS
# ==================================================

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Isolation Forest",
    iforest_count
)

col2.metric(
    "One-Class SVM",
    svm_count
)

col3.metric(
    "Autoencoder",
    ae_count
)

col4.metric(
    "Consensus",
    consensus_count
)


# ==================================================
# MODEL SELECTION
# ==================================================

if model == "Isolation Forest":

    selected_df = iforest_df

    anomaly_col = (
        "isolationforest_anomaly"
    )

    score_col = (
        "isolationforest_score"
    )

elif model == "One-Class SVM":

    selected_df = svm_df

    anomaly_col = (
        "oneclasssvm_anomaly"
    )

    score_col = (
        "oneclasssvm_score"
    )

else:

    selected_df = ae_df

    anomaly_col = (
        "autoencoder_anomaly"
    )

    score_col = (
        "autoencoder_score"
    )


# ==================================================
# CHART
# ==================================================

st.subheader(
    f"{ticker} - {model}"
)

fig = create_anomaly_chart(
    selected_df,
    anomaly_col,
    score_col,
    f"{ticker} - {model}"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# MODEL ANOMALIES
# ==================================================

st.subheader(
    f"{model} Anomalies"
)

model_anomalies = selected_df[
    selected_df[
        anomaly_col
    ]
]

st.dataframe(
    model_anomalies[
        [
            "Date",
            "Close",
            score_col
        ]
    ],
    use_container_width=True
)

# ==========================================
# ANOMALY EXPLANATION
# ==========================================

if len(model_anomalies) > 0:

    st.subheader(
        "Anomaly Explanation"
    )

    selected_date = st.selectbox(
        "Select anomaly date",
        model_anomalies["Date"]
    )

    selected_row = model_anomalies[
        model_anomalies["Date"] == selected_date
    ].iloc[0]

    reasons = explain_anomaly(
        selected_row
    )

    if len(reasons) > 0:

        for reason in reasons:

            st.write(
                f"• {reason}"
            )

    else:

        st.info(
            "No specific explanation rules were triggered."
        )


# ==================================================
# CONSENSUS ANOMALIES
# ==================================================

st.subheader(
    "Consensus Anomalies"
)

st.dataframe(
    consensus_df,
    use_container_width=True
)


# ==================================================
# AUTOENCODER LOSS
# ==================================================

if model == "Autoencoder":

    st.subheader(
        "Training Loss"
    )

    st.image(
        f"reports/figures/{ticker}_ae_loss.png",
        width=700
    )


# ==================================================
# REPORT
# ==================================================

try:

    st.subheader(
        "Generated Report"
    )

    with open(
        f"reports/results/{ticker}_report.txt",
        "r"
    ) as f:

        report = f.read()

    st.text(report)

except FileNotFoundError:

    st.warning(
        "Report file not found."
    )

# ==================================================
# Download buttons
# ==================================================

st.download_button(
    "Download Consensus CSV",
    consensus_df.to_csv(index=False),
    file_name=f"{ticker}_consensus.csv"
)


