"""Map toxicity probability to risk bucket and moderator-assist action.

Thresholds (human-in-the-loop; moderator is final authority):
- prob < 0.50 => low / no_action
- 0.50 <= prob < 0.85 => medium / low_priority_queue
- 0.85 <= prob < 0.95 => high / high_priority_queue
- prob >= 0.95 => critical / highest_priority_queue
"""

from app.schemas import ActionRecommendation, RiskBucket


def probability_to_risk_and_action(probability: float) -> tuple[RiskBucket, ActionRecommendation]:
    if probability < 0.50:
        return "low", "no_action"
    if probability < 0.85:
        return "medium", "low_priority_queue"
    if probability < 0.95:
        return "high", "high_priority_queue"
    return "critical", "highest_priority_queue"
