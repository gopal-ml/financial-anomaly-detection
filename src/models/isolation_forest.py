from sklearn.ensemble import IsolationForest
import joblib

from configs.settings import RANDOM_STATE
from src.models.base_detector import BaseDetector


class IsolationForestDetector(BaseDetector):
    def __init__(
        self,
        n_estimators=200,
        contamination=0.01,
        random_state=RANDOM_STATE
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)

        return self

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds == -1)

    def anomaly_score(self, X):
        return self.model.decision_function(X)
    
    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
