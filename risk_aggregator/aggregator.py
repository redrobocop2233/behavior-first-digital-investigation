"""Combines signals into final risk score."""

import pandas as pd
from typing import Dict, Optional


class RiskAggregator:
    """
    Combines signals into final risk score:
    R_i = w1*s(x_i) + w2*C_i + w3*Cluster_i
    Weights are regulator-defined and auditable.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        # Default weights (would be set by regulators)
        self.weights = weights or {
            'anomaly_weight': 0.5,
            'centrality_weight': 0.2,
            'cluster_weight': 0.3
        }
        self.threshold = 0.7  # τ - risk threshold for legal review

    def calculate_risk_scores(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate aggregated risk scores for all entities.
        """
        # Normalize centrality
        max_centrality = features_df['centrality'].max()
        norm_centrality = features_df['centrality'] / max_centrality if max_centrality > 0 else 0

        # Normalize cluster impact (larger clusters = higher risk)
        max_cluster_size = features_df['cluster_size'].max()
        norm_cluster = features_df['cluster_size'] / max_cluster_size if max_cluster_size > 0 else 0

        # Calculate risk score
        risk_score = (
            self.weights['anomaly_weight'] * features_df['anomaly_score'] +
            self.weights['centrality_weight'] * norm_centrality +
            self.weights['cluster_weight'] * norm_cluster
        )

        result_df = features_df.copy()
        result_df['risk_score'] = risk_score
        result_df['needs_review'] = (risk_score > self.threshold).astype(int)

        return result_df