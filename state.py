from typing import TypedDict, Any


class MarketingState(TypedDict):
    vehicle: dict
    marketing_copy: str
    review_feedback: Any
    quality_score: int
    approved: bool
    iteration: int