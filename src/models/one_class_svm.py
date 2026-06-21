from sklearn.svm import OneClassSVM
import joblib

from src.models.base_detector import BaseDetector

class OneClassSVMDetector(BaseDetector):

    def __init__(
        self,
        kernel="rbf",
        nu=0.01,
        gamma="scale"
    ):

        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
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
        return self

    def load(self, path):
        self.model = joblib.load( path)
