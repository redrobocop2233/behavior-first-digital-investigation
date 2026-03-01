"""Generate visualizations and reports for the investigation."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from typing import List, Dict
import os


class InvestigationReporter:
    """Generate visualizations and reports for the investigation."""

    def __init__(self, output_dir: str = './reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_risk_distribution(self, results_df: pd.DataFrame):
        """Plot the distribution of risk scores."""
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(results_df['risk_score'], bins=30, edgecolor='black', alpha=0.7)
        plt.axvline(x=0.7, color='red', linestyle='--', label='Review Threshold (τ=0.7)')
        plt.xlabel('Risk Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Risk Scores')
        plt.legend()

        plt.subplot(1, 2, 2)
        risk_bins = pd.cut(results_df['risk_score'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
        risk_counts = risk_bins.value_counts().sort_index()
        plt.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%')
        plt.title('Risk Score Categories')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/risk_distribution.png')
        plt.show()

    def plot_feature_importance(self, results_df: pd.DataFrame):
        """Visualize feature importance in anomaly detection."""
        features = ['frequency', 'behavioral_drift', 'reaction_speed', 'variance', 'centrality']

        # Compare means between normal and anomalous
        normal = results_df[results_df['is_anomaly'] == 0][features].mean()
        anomalous = results_df[results_df['is_anomaly'] == 1][features].mean()

        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, normal.values, width, label='Normal', alpha=0.8)
        ax.bar(x + width/2, anomalous.values, width, label='Anomalous', alpha=0.8)

        ax.set_xlabel('Features')
        ax.set_ylabel('Mean Value (scaled)')
        ax.set_title('Feature Comparison: Normal vs Anomalous Entities')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/feature_importance.png')
        plt.show()

    def plot_coordination_network(self, results_df: pd.DataFrame, top_n: int = 20):
        """Visualize coordination clusters."""
        # Get top risky entities
        top_entities = results_df.nlargest(top_n, 'risk_score')

        # Create network graph
        G = nx.Graph()

        # Add nodes with risk scores
        for idx, row in top_entities.iterrows():
            G.add_node(row['entity_id'],
                      risk=row['risk_score'],
                      cluster=row['cluster_id'] if row['cluster_id'] >= 0 else 'none')

        # Connect entities in same cluster
        clusters = top_entities[top_entities['cluster_id'] >= 0].groupby('cluster_id')
        for cluster_id, group in clusters:
            entities = group['entity_id'].tolist()
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    G.add_edge(entities[i], entities[j], weight=group['cluster_size'].iloc[0])

        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Color nodes by cluster
        colors = [G.nodes[node]['cluster'] for node in G.nodes()]
        # Convert to numeric for colormap
        color_map = {'none': 0}
        for i, c in enumerate(set(colors) - {'none'}):
            color_map[c] = i + 1
        numeric_colors = [color_map[c] for c in colors]

        # Size nodes by risk
        sizes = [G.nodes[node]['risk'] * 1000 for node in G.nodes()]

        nx.draw(G, pos,
                node_color=numeric_colors,
                node_size=sizes,
                with_labels=True,
                font_size=8,
                cmap='viridis',
                edge_color='gray',
                alpha=0.7)

        plt.title('Coordination Network of High-Risk Entities')
        plt.savefig(f'{self.output_dir}/coordination_network.png')
        plt.show()

    def generate_report(self, results_df: pd.DataFrame, firewall_log: List[Dict]) -> str:
        """Generate a comprehensive investigation report."""

        report = []
        report.append("=" * 80)
        report.append("BEHAVIOR-FIRST DIGITAL INVESTIGATION REPORT")
        report.append("=" * 80)
        report.append(f"\nReport Generated: {datetime.now()}")
        report.append(f"\nTotal Entities Analyzed: {len(results_df)}")
        report.append(f"Total Events Processed: {results_df['total_events'].sum()}")

        # Risk statistics
        report.append("\n" + "-" * 40)
        report.append("RISK ANALYSIS")
        report.append("-" * 40)

        high_risk = len(results_df[results_df['risk_score'] > 0.7])
        medium_risk = len(results_df[(results_df['risk_score'] > 0.4) & (results_df['risk_score'] <= 0.7)])
        low_risk = len(results_df[results_df['risk_score'] <= 0.4])

        report.append(f"High Risk (Requires Review): {high_risk}")
        report.append(f"Medium Risk: {medium_risk}")
        report.append(f"Low Risk: {low_risk}")

        # Anomaly statistics
        report.append("\n" + "-" * 40)
        report.append("ANOMALY DETECTION")
        report.append("-" * 40)

        anomalies = results_df[results_df['is_anomaly'] == 1]
        report.append(f"Entities with Anomalous Behavior: {len(anomalies)}")
        if len(anomalies) > 0:
            report.append(f"Average Risk Score of Anomalies: {anomalies['risk_score'].mean():.3f}")

        # Cluster statistics
        report.append("\n" + "-" * 40)
        report.append("COORDINATION DETECTION")
        report.append("-" * 40)

        clustered = results_df[results_df['is_clustered'] == 1]
        num_clusters = results_df[results_df['cluster_id'] >= 0]['cluster_id'].nunique()
        report.append(f"Entities in Coordinated Groups: {len(clustered)}")
        report.append(f"Number of Coordination Clusters: {num_clusters}")

        if num_clusters > 0:
            largest_cluster = results_df.loc[results_df['cluster_size'].idxmax()]
            report.append(f"Largest Cluster Size: {largest_cluster['cluster_size']}")

        # Firewall activity
        report.append("\n" + "-" * 40)
        report.append("ETHICAL FIREWALL ACTIVITY")
        report.append("-" * 40)

        reveals = [log for log in firewall_log if log['action'] == 'reveal']
        attempts = [log for log in firewall_log if log['action'] == 'attempt']
        forgetting = [log for log in firewall_log if log['action'] == 'forgetting']

        report.append(f"Identity Reveal Requests: {len(attempts)}")
        report.append(f"Court-Approved Reveals: {len(reveals)}")
        report.append(f"Automatic Forgetting Events: {len(forgetting)}")

        # Top high-risk entities (anonymized)
        report.append("\n" + "-" * 40)
        report.append("TOP HIGH-RISK ENTITIES (ANONYMIZED)")
        report.append("-" * 40)

        top_risky = results_df.nlargest(5, 'risk_score')[['entity_id', 'risk_score', 'is_anomaly', 'cluster_id']]
        for idx, row in top_risky.iterrows():
            report.append(f"\nEntity: {row['entity_id']}")
            report.append(f"  Risk Score: {row['risk_score']:.3f}")
            report.append(f"  Anomalous: {'YES' if row['is_anomaly'] else 'NO'}")
            report.append(f"  Cluster: {row['cluster_id'] if row['cluster_id'] >= 0 else 'None'}")

        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        # Save report
        report_text = '\n'.join(report)
        with open(f'{self.output_dir}/investigation_report.txt', 'w') as f:
            f.write(report_text)

        return report_text