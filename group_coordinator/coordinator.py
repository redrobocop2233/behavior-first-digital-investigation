"""Detects coordinated behavior clusters using DBSCAN."""

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class GroupCoordinator:
    """
    Detects coordinated behavior clusters using DBSCAN.
    Condition: |{xj: ||xi - xj|| ≤ ε}| ≥ MinPts
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 3):
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.scaler = StandardScaler()

    def detect_clusters(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify dense abnormal behavioral clusters.
        """
        feature_columns = ['frequency', 'behavioral_drift', 'variance', 'anomaly_score']
        X = features_df[feature_columns].fillna(0)

        # Standardize
        X_scaled = self.scaler.fit_transform(X)

        # Fit DBSCAN
        cluster_labels = self.dbscan.fit_predict(X_scaled)

        # Add to dataframe
        result_df = features_df.copy()
        result_df['cluster_id'] = cluster_labels
        result_df['is_clustered'] = (cluster_labels != -1).astype(int)

        # Calculate cluster sizes
        cluster_sizes = result_df['cluster_id'].value_counts().to_dict()
        result_df['cluster_size'] = result_df['cluster_id'].map(cluster_sizes).fillna(1)

        return result_df