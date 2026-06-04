from typing import TypedDict, Any


class MarketingState(TypedDict):
    vehicle: dict          # All parsed VPIC + DMS data + dealer info
    marketing_copy: str    # Generated/refined copy
    review_feedback: Any   # Dict from reviewer node
    quality_score: int     # 0-100
    approved: bool
    iteration: int
    max_iterations: int