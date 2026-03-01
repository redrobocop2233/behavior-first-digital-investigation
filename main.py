"""Main entry point for the Behavior-First Investigation System."""

import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from graph_builder import EventGraphBuilder
from feature_extractor import FeatureExtractor
from anomaly_detector import AnomalyDetector
from group_coordinator import GroupCoordinator
from risk_aggregator import RiskAggregator
from ethical_firewall import EthicalFirewall
from reporter import InvestigationReporter


class BehaviorFirstInvestigationSystem:
    """
    Complete end-to-end behavior-first investigation pipeline.
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.event_graph = None
        self.features_df = None
        self.anomaly_results = None
        self.cluster_results = None
        self.risk_results = None
        self.firewall = None
        self.reporter = InvestigationReporter()

    def run_investigation(self, forensics_path: str, payment_path: str, cyber_path: str):
        """Execute the complete investigation pipeline."""

        print("=" * 60)
        print("BEHAVIOR-FIRST INVESTIGATION SYSTEM")
        print("=" * 60)

        # Step 1: Load and anonymize data
        print("\n[1/7] Loading and anonymizing data...")
        unified_data = self.data_loader.prepare_unified_data(
            forensics_path, payment_path, cyber_path
        )
        print(f"Loaded {len(unified_data)} events from {unified_data['dataset'].nunique()} datasets")
        print(f"Anonymized {unified_data['entity_id'].nunique()} unique entities")

        # Step 2: Build event graph
        print("\n[2/7] Building abstract event graph...")
        graph_builder = EventGraphBuilder()
        self.event_graph = graph_builder.build_from_dataframe(unified_data)
        print(f"Graph constructed: {self.event_graph.number_of_nodes()} nodes, "
              f"{self.event_graph.number_of_edges()} edges")

        # Step 3: Extract features
        print("\n[3/7] Extracting behavioral features...")
        feature_extractor = FeatureExtractor(self.event_graph)
        self.features_df = feature_extractor.extract_all_features()
        print(f"Extracted features for {len(self.features_df)} entities")

        # Step 4: Detect anomalies
        print("\n[4/7] Detecting statistical anomalies...")
        anomaly_detector = AnomalyDetector(contamination=0.15)
        self.anomaly_results = anomaly_detector.fit_and_predict(self.features_df)
        anomaly_count = self.anomaly_results['is_anomaly'].sum()
        print(f"Detected {anomaly_count} entities with anomalous behavior "
              f"({anomaly_count/len(self.anomaly_results)*100:.1f}%)")

        # Step 5: Detect coordination
        print("\n[5/7] Detecting coordinated groups...")
        coordinator = GroupCoordinator(eps=0.6, min_samples=3)
        self.cluster_results = coordinator.detect_clusters(self.anomaly_results)
        clustered_count = self.cluster_results['is_clustered'].sum()
        print(f"Found {clustered_count} entities in coordinated clusters")

        # Step 6: Calculate risk scores
        print("\n[6/7] Aggregating risk scores...")
        risk_aggregator = RiskAggregator()
        self.risk_results = risk_aggregator.calculate_risk_scores(self.cluster_results)
        review_needed = self.risk_results['needs_review'].sum()
        print(f"{review_needed} entities require legal review (risk score > 0.7)")

        # Step 7: Initialize ethical firewall
        print("\n[7/7] Initializing ethical firewall...")
        self.firewall = EthicalFirewall(self.data_loader.anonymized_entities)

        # Simulate some court approvals for top risky entities
        top_risky = self.risk_results.nlargest(3, 'risk_score')['entity_id'].tolist()
        for i, entity in enumerate(top_risky):
            self.firewall.court_approve_reveal(entity, f"CASE-2024-{i+100}")
        print(f"Simulated court approval for {len(top_risky)} entities")

        print("\n" + "=" * 60)
        print("INVESTIGATION COMPLETE")
        print("=" * 60)

        return self.risk_results

    def demonstrate_ethical_firewall(self):
        """Demonstrate how the ethical firewall protects privacy."""

        print("\n" + "=" * 60)
        print("ETHICAL FIREWALL DEMONSTRATION")
        print("=" * 60)

        # Select a high-risk entity
        high_risk_entities = self.risk_results.nlargest(5, 'risk_score')

        for idx, row in high_risk_entities.iterrows():
            entity_id = row['entity_id']
            risk_score = row['risk_score']

            print(f"\n--- Entity: {entity_id} (Risk Score: {risk_score:.3f}) ---")

            # Attempt to reveal identity without court approval
            print("\n[1] Investigator requests identity reveal (NO court approval):")
            response1 = self.firewall.request_identity_reveal(entity_id, risk_score)
            print(f"    → {response1['message']}")

            # Get court approval
            print(f"\n[2] Court approves reveal for this entity:")
            self.firewall.court_approve_reveal(entity_id, "CASE-2024-APPROVED")
            print(f"    → Court approval granted")

            # Attempt again with approval
            print(f"\n[3] Investigator requests identity reveal (WITH court approval):")
            response2 = self.firewall.request_identity_reveal(entity_id, risk_score)
            if response2['identity_revealed']:
                print(f"    → Identity REVEALED: {response2['original_id']}")
            else:
                print(f"    → {response2['message']}")

            # Simulate acquittal and forgetting
            if risk_score > 0.8:
                print(f"\n[4] Entity acquitted - automatic forgetting triggered:")
                forget_response = self.firewall.automatic_forgetting(entity_id, 'acquitted')
                print(f"    → {forget_response['message']}")

        # Show audit log
        print("\n" + "-" * 40)
        print("FIREWALL AUDIT LOG")
        print("-" * 40)
        for entry in self.firewall.get_audit_log()[-6:]:  # Show last 6 entries
            print(f"{entry['timestamp'].strftime('%H:%M:%S')} - {entry['action']}: {entry['anonymized_id']}")

    def generate_visualizations(self):
        """Generate all visualizations."""
        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        self.reporter.plot_risk_distribution(self.risk_results)
        self.reporter.plot_feature_importance(self.risk_results)
        self.reporter.plot_coordination_network(self.risk_results)

        report = self.reporter.generate_report(self.risk_results, self.firewall.forgetting_log)
        print("\nReport saved to ./reports/investigation_report.txt")

    def analyze_specific_patterns(self):
        """Analyze specific behavioral patterns of interest."""

        print("\n" + "=" * 60)
        print("BEHAVIORAL PATTERN ANALYSIS")
        print("=" * 60)

        # Find entities with high behavioral drift
        high_drift = self.risk_results.nlargest(5, 'behavioral_drift')
        print("\nEntities with highest behavioral drift (significant behavior change):")
        for idx, row in high_drift.iterrows():
            print(f"  {row['entity_id']}: drift={row['behavioral_drift']:.3f}, "
                  f"risk={row['risk_score']:.3f}")

        # Find entities with unusual reaction speeds
        fast_reactors = self.risk_results.nlargest(5, 'reaction_speed')
        print("\nEntities with fastest reaction speeds:")
        for idx, row in fast_reactors.iterrows():
            print(f"  {row['entity_id']}: speed={row['reaction_speed']:.3f}, "
                  f"risk={row['risk_score']:.3f}")

        # Find central entities in coordination networks
        high_centrality = self.risk_results.nlargest(5, 'centrality')
        print("\nMost central entities (high connectivity):")
        for idx, row in high_centrality.iterrows():
            print(f"  {row['entity_id']}: centrality={row['centrality']}, "
                  f"risk={row['risk_score']:.3f}")

        # Analyze coordination clusters
        clusters = self.risk_results[self.risk_results['cluster_id'] >= 0]
        if len(clusters) > 0:
            print("\nCoordination clusters detected:")
            for cluster_id in clusters['cluster_id'].unique():
                cluster_members = clusters[clusters['cluster_id'] == cluster_id]
                avg_risk = cluster_members['risk_score'].mean()
                print(f"  Cluster {cluster_id}: {len(cluster_members)} members, "
                      f"avg risk={avg_risk:.3f}")


def main():
    """Main entry point for the application."""
    investigator = BehaviorFirstInvestigationSystem()
    
    # Update these paths to your actual data paths
    results = investigator.run_investigation(
        forensics_path="data/forensics.csv",
        payment_path="data/payments.csv",
        cyber_path="data/cyber.csv"
    )
    
    investigator.generate_visualizations()
    investigator.demonstrate_ethical_firewall()
    investigator.analyze_specific_patterns()
    
    return investigator


if __name__ == "__main__":
    main()