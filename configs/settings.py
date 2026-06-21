from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = (
    DATA_DIR /
    "processed"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models"
)

REPORT_DIR = (
    PROJECT_ROOT /
    "reports"
)

RANDOM_STATE = 42

TRAIN_RATIO = 0.8

MODEL_NAMES = {
    "isolationforest": "Isolation Forest",
    "oneclasssvm": "One-Class SVM",
    "autoencoder": "Autoencoder"
}

IFOREST_ESTIMATORS = 200
IFOREST_CONTAMINATION = 0.01

SVM_NU = 0.001
SVM_GAMMA = "scale"

AE_EPOCHS = 100
AE_BATCH_SIZE = 64
AE_LR = 1e-3

ANOMALY_PERCENTILE = 98.5
