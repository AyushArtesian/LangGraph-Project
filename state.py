from typing import TypedDict


class MarketingState(TypedDict):
    car_json: dict

    marketing_copy: str

    review_feedback: str

    quality_score: int

    approved: bool

    iteration: int