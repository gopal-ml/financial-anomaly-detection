from src.pipelines.run_all_models import (
    run_all_models
)

tickers = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA"
]

for ticker in tickers:

    run_all_models(
        ticker=ticker
    )
