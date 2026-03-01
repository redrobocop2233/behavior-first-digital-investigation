"""Detects statistically abnormal behavior using Isolation Forest."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Detects statistically abnormal behavior using Isolation Forest.
    Anomaly score s(x) = 2^(-E[h(x)]/c(n))
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )
        self.scaler = StandardScaler()
        self.feature_columns = ['frequency', 'behavioral_drift', 'reaction_speed', 'variance', 'centrality']

    def fit_and_predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit Isolation Forest and compute anomaly scores.
        s(x) = 2^(-E[h(x)]/c(n)) where shorter path = more anomalous
        """
        # Extract features
        X = features_df[self.feature_columns].fillna(0)

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Fit model and get scores
        self.model.fit(X_scaled)

        # Get anomaly scores (more negative = more anomalous)
        raw_scores = self.model.score_samples(X_scaled)

        # Transform to [0,1] where 1 is most anomalous
        # s(x) = 2^(-E[h(x)]/c(n)) - we'll normalize to 0-1
        normalized_scores = 1 / (1 + np.exp(-raw_scores))  # sigmoid transformation

        # Get predictions (-1 for anomalies, 1 for normal)
        predictions = self.model.predict(X_scaled)

        # Add to dataframe
        result_df = features_df.copy()
        result_df['anomaly_score'] = normalized_scores
        result_df['is_anomaly'] = (predictions == -1).astype(int)

        return result_df