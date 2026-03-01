"""Transforms entity behavior into measurable geometry."""

import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class FeatureExtractor:
    """
    Transforms entity behavior into measurable geometry.
    x_i = [f_i, Δ_i, v_i, σ_i, C_i]
    """

    def __init__(self, event_graph: nx.MultiDiGraph, time_interval: str = '1H'):
        self.graph = event_graph
        self.time_interval = time_interval
        self.features = {}

    def extract_all_features(self) -> pd.DataFrame:
        """Extract feature vectors for all entities."""

        entities = set(self.graph.nodes())
        feature_data = []

        for entity in entities:
            features = self.extract_entity_features(entity)
            if features:
                feature_data.append({
                    'entity_id': entity,
                    **features
                })

        return pd.DataFrame(feature_data)

    def extract_entity_features(self, entity_id: str) -> Dict[str, float]:
        """
        Extract the 5 key features for an entity:
        f_i: frequency of activity
        Δ_i: behavioral drift
        v_i: reaction speed
        σ_i: variance in behavior
        C_i: network centrality
        """
        # Get all events for this entity
        events = self._get_entity_events(entity_id)
        if len(events) < 2:
            return None

        # 1. Frequency (f_i = N_i / T)
        time_span = (max(e['timestamp'] for e in events) -
                     min(e['timestamp'] for e in events)).total_seconds() / 3600  # hours
        frequency = len(events) / max(time_span, 1)  # events per hour

        # 2. Behavioral drift (Δ_i = ||μ_current - μ_past||)
        # Split events into two halves
        mid_point = len(events) // 2
        past_events = events[:mid_point]
        current_events = events[mid_point:]

        past_pattern = self._create_behavior_pattern(past_events)
        current_pattern = self._create_behavior_pattern(current_events)

        drift = np.linalg.norm(np.array(current_pattern) - np.array(past_pattern))

        # 3. Reaction speed (v_i = 1/(t_action - t_trigger))
        reaction_speeds = self._calculate_reaction_speeds(events)
        avg_reaction_speed = np.mean(reaction_speeds) if reaction_speeds else 0

        # 4. Variance (σ_i²)
        # Calculate variance in event intervals
        intervals = [(events[i+1]['timestamp'] - events[i]['timestamp']).total_seconds()
                    for i in range(len(events)-1)]
        variance = np.var(intervals) if intervals else 0

        # 5. Centrality (C_i = deg(i))
        centrality = self.graph.degree(entity_id)

        return {
            'frequency': frequency,
            'behavioral_drift': drift,
            'reaction_speed': avg_reaction_speed,
            'variance': variance,
            'centrality': centrality,
            'total_events': len(events)
        }

    def _get_entity_events(self, entity_id: str) -> List[Dict]:
        """Get all events for an entity."""
        events = []

        # Outgoing events
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            events.append({
                'timestamp': data['timestamp'],
                'event_type': data['event_type'],
                'direction': 'out',
                'target': target
            })

        # Incoming events
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            events.append({
                'timestamp': data['timestamp'],
                'event_type': data['event_type'],
                'direction': 'in',
                'source': source
            })

        return sorted(events, key=lambda x: x['timestamp'])

    def _create_behavior_pattern(self, events: List[Dict]) -> List[float]:
        """Create a numerical pattern of behavior for an entity."""
        if not events:
            return [0, 0, 0]

        # Simple pattern: [activity_rate, diversity, reciprocity]
        activity_rate = len(events) / 24  # normalize to daily rate

        event_types = set(e['event_type'] for e in events)
        diversity = len(event_types) / 10  # normalize

        reciprocity = sum(1 for e in events if e['direction'] == 'in') / max(len(events), 1)

        return [activity_rate, diversity, reciprocity]

    def _calculate_reaction_speeds(self, events: List[Dict]) -> List[float]:
        """Calculate reaction speeds to external events."""
        speeds = []

        for i in range(1, len(events)):
            # Look for pairs where incoming event triggers outgoing event
            if events[i-1]['direction'] == 'in' and events[i]['direction'] == 'out':
                reaction_time = (events[i]['timestamp'] - events[i-1]['timestamp']).total_seconds()
                if reaction_time > 0:
                    speeds.append(1.0 / reaction_time)

        return speeds