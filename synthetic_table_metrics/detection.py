"""detection.py

How dificult is it to separate the synthetic dataset from the original
dataset using machine learning?

- Transform synthetic and real using a generic transformer: the HyperTransformer.
- Scale the data using a generic scaler: RobustScaler.
- Concatenate the datasets and create a label array (real: 1, synthetic: 0).
- Train a simple machine learning model to separate the data.
- The higher the auroc, the worse the synthetic data.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC


# Possible models to try to separate real and synthetic datasets
MODELS = {
    "logistic": LogisticRegression(solver="liblinear"),
    "SVC": SVC(probability=True, gamma="scale"),
}


class Detection:
    """
    Calculate how easy a machine learning model can separate the real
    from the synthetic data. If it is easy to separate, the synthetic
    data is bad.
    """

    def __init__(self, models=["logistic", "SVC"]):
        self.models = models  # What models to use

    @staticmethod
    def create_labeled_dataset(data):
        """
        Concatenate the real and synthesized dataset. Then create a label and
        replace NaN's.
        """
        # Append both datasets and label real and synthetic in y column
        X = np.concatenate([data.real, data.synthetic])
        y = np.hstack([np.ones(len(data.real)), np.zeros(len(data.synthetic))])
        X[np.isnan(X)] = 0.0  # NaN's will become 0.0
        return X, y

    @staticmethod
    def create_model(model_name):
        """
        Create a model Pipeline with a scaler and a model picked
        based on model_name.
        """
        # Model pipeline
        return Pipeline(
            [  # Scale everything generally
                ("scaler", RobustScaler()),
                # Simple logistic regression
                ("classifier", MODELS[model_name]),
            ]
        )

    def calculate_detectability(self, X, y):
        """
        Train models to separate the real and synthetic data. Then
        calculate the AUROC for each model and return the average of
        the AUROC scores.
        """
        # Fit the model multiple times and save the scores
        scores = []
        for model_name in self.models:
            model = self.create_model(model_name)
            kf = StratifiedKFold(n_splits=3, shuffle=True)
            for train_index, test_index in kf.split(X, y):
                # Fit the model
                model.fit(X[train_index], y[train_index])
                # Create predictions
                y_pred = model.predict_proba(X[test_index])[:, 1]
                # Calculate area under recieving operating characteristics
                auroc = roc_auc_score(y[test_index], y_pred)
                if auroc < 0.5:
                    auroc = 1 - auroc
                # Save score for fold
                scores.append(auroc)
        # Print result
        return np.mean(scores)

    def run(self, data):
        """
        Concatenate the data and create a label column. Then calculate
        the AUROC value and return a scaled value.
        """
        X, y = self.create_labeled_dataset(data)
        auroc = self.calculate_detectability(X, y)
        return (auroc - 0.5) * 2  # Detection value
