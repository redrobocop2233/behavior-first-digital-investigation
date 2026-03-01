"""Load and preprocess the three datasets into a unified format."""

import pandas as pd
import hashlib
from typing import Dict


class DataLoader:
    """Load and preprocess the three datasets into a unified format."""

    def __init__(self):
        self.datasets = {}
        self.event_graph = None
        self.anonymized_entities = {}

    def load_synthetic_forensics(self, path: str) -> pd.DataFrame:
        """Load Synthetic Digital Forensics Dataset."""
        df = pd.read_csv(path)
        # Standardize column names
        df = df.rename(columns={
            'Timestamp': 'timestamp',
            'User_ID': 'entity_id',
            'Activity_Type': 'event_type',
            'Device_Type': 'device_info'
        })
        df['dataset'] = 'forensics'
        return df

    def load_payment_fraud(self, path: str) -> pd.DataFrame:
        """Load Digital Payment Fraud Detection Dataset."""
        df = pd.read_csv(path)
        df = df.rename(columns={
            'transaction_date': 'timestamp',
            'user_id': 'entity_id',
            'transaction_type': 'event_type',
            'amount': 'value'
        })
        df['dataset'] = 'payment'
        return df

    def load_cybersecurity(self, path: str) -> pd.DataFrame:
        """Load Cybersecurity Attack Dataset."""
        df = pd.read_csv(path)
        df = df.rename(columns={
            'timestamp': 'timestamp',
            'source_ip': 'entity_id',
            'attack_type': 'event_type',
            'protocol': 'device_info'
        })
        df['dataset'] = 'cybersecurity'
        return df

    def anonymize_entity(self, entity_id: str) -> str:
        """Hash entity IDs to preserve privacy during initial analysis."""
        if entity_id not in self.anonymized_entities:
            # Use SHA-256 for anonymization
            hash_obj = hashlib.sha256(str(entity_id).encode())
            self.anonymized_entities[entity_id] = hash_obj.hexdigest()[:16]
        return self.anonymized_entities[entity_id]

    def prepare_unified_data(self, forensics_path: str, payment_path: str, cyber_path: str) -> pd.DataFrame:
        """Load and combine all datasets with anonymization."""

        # Load individual datasets
        df1 = self.load_synthetic_forensics(forensics_path)
        df2 = self.load_payment_fraud(payment_path)
        df3 = self.load_cybersecurity(cyber_path)

        # Combine
        unified_df = pd.concat([df1, df2, df3], ignore_index=True)

        # Convert timestamp to datetime
        unified_df['timestamp'] = pd.to_datetime(unified_df['timestamp'])

        # Anonymize entity IDs
        unified_df['anonymized_id'] = unified_df['entity_id'].apply(self.anonymize_entity)

        # Sort by timestamp
        unified_df = unified_df.sort_values('timestamp').reset_index(drop=True)

        return unified_df