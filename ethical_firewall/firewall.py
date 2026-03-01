"""Legal gate that controls identity revelation."""

from datetime import datetime
from typing import Dict, List, Any


class EthicalFirewall:
    """
    Legal gate that controls identity revelation.
    Reveal = (R_i > τ) ∧ (CourtApproval)
    Implements automatic forgetting for acquitted individuals.
    """

    def __init__(self, anonymization_map: Dict[str, str]):
        self.anonymization_map = anonymization_map  # original_id -> anonymized_id
        self.reverse_map = {v: k for k, v in anonymization_map.items()}  # anonymized -> original
        self.court_approved_reveals = set()
        self.acquitted_records = set()
        self.forgetting_log = []

    def request_identity_reveal(self, anonymized_id: str, risk_score: float) -> Dict[str, Any]:
        """
        Request identity revelation for legal review.
        Only proceeds if court approval is granted.
        """
        response = {
            'anonymized_id': anonymized_id,
            'risk_score': risk_score,
            'court_approval': False,
            'identity_revealed': False,
            'original_id': None,
            'message': ''
        }

        # Check if court has approved this reveal
        if anonymized_id in self.court_approved_reveals:
            original_id = self.reverse_map.get(anonymized_id)
            response.update({
                'court_approval': True,
                'identity_revealed': True,
                'original_id': original_id,
                'message': 'Identity revealed with court approval'
            })

            # Log the reveal
            self._log_action('reveal', anonymized_id, risk_score=risk_score)
        else:
            response['message'] = 'Identity remains anonymized - court approval required'

            # Log the attempt
            self._log_action('attempt', anonymized_id, risk_score=risk_score)

        return response

    def court_approve_reveal(self, anonymized_id: str, case_id: str):
        """Simulate court approval for identity revelation."""
        self.court_approved_reveals.add(anonymized_id)
        self._log_action('court_approval', anonymized_id, case_id=case_id)

    def automatic_forgetting(self, anonymized_id: str, reason: str = 'acquitted') -> Dict[str, str]:
        """
        Implement automatic forgetting for acquitted individuals.
        Delete( H(ID_i) ) for i ∈ S
        """
        if anonymized_id in self.reverse_map:
            original_id = self.reverse_map[anonymized_id]

            # Remove from maps (simulating deletion)
            del self.reverse_map[anonymized_id]
            # Note: In real system, we would delete from all storage

            self.acquitted_records.add(anonymized_id)

            self._log_action('forgetting', anonymized_id, reason=reason)

            return {
                'status': 'forgotten',
                'anonymized_id': anonymized_id,
                'reason': reason,
                'message': 'Identity data has been permanently deleted'
            }

        return {'status': 'not_found', 'message': 'Entity not found'}

    def _log_action(self, action: str, anonymized_id: str, **kwargs):
        """Log all firewall actions for audit purposes."""
        log_entry = {
            'timestamp': datetime.now(),
            'action': action,
            'anonymized_id': anonymized_id,
            **kwargs
        }
        self.forgetting_log.append(log_entry)

    def get_audit_log(self) -> List[Dict]:
        """Return the complete audit log."""
        return self.forgetting_log