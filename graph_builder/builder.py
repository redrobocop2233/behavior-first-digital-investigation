"""Builds an abstract event graph from anonymized data."""

import networkx as nx
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple


class EventGraphBuilder:
    """
    Builds an abstract event graph from anonymized data.
    G = (V, E) where vertices are entities and edges represent interactions.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.event_types = set()

    def build_from_dataframe(self, df: pd.DataFrame) -> nx.MultiDiGraph:
        """
        Convert raw logs into abstract events e = (v_i, v_j, t, φ)
        """
        for idx, row in df.iterrows():
            # Create abstract event
            event = {
                'source': row['anonymized_id'],
                'target': row.get('target_id', row['anonymized_id']),  # Self if no target
                'timestamp': row['timestamp'],
                'event_type': row['event_type'],
                'metadata': {
                    'dataset': row['dataset'],
                    'value': row.get('value', 0)
                }
            }

            # Add to graph as edge
            self.graph.add_edge(
                event['source'],
                event['target'],
                timestamp=event['timestamp'],
                event_type=event['event_type'],
                metadata=event['metadata']
            )

            self.event_types.add(event['event_type'])

        return self.graph

    def get_entity_events(self, entity_id: str, time_window: Optional[Tuple[datetime, datetime]] = None) -> List[Dict]:
        """Get all events for a specific entity within optional time window."""
        events = []

        # Check outgoing edges
        for _, target, data in self.graph.out_edges(entity_id, data=True):
            if time_window:
                if time_window[0] <= data['timestamp'] <= time_window[1]:
                    events.append({
                        'type': 'outgoing',
                        'target': target,
                        **data
                    })
            else:
                events.append({
                    'type': 'outgoing',
                    'target': target,
                    **data
                })

        # Check incoming edges
        for source, _, data in self.graph.in_edges(entity_id, data=True):
            if time_window:
                if time_window[0] <= data['timestamp'] <= time_window[1]:
                    events.append({
                        'type': 'incoming',
                        'source': source,
                        **data
                    })
            else:
                events.append({
                    'type': 'incoming',
                    'source': source,
                    **data
                })

        return sorted(events, key=lambda x: x['timestamp'])