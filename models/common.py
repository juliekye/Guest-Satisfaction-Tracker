from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from db.tables import Review


class SentimentType(Enum):
    POSITIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2

@dataclass
class AugmentedReview:
    review: Review
    assigned_topics: list[str]
    sentiment_type: SentimentType
    review_date: datetime


@dataclass
class AugmentedTopic:
    topic: str
    sentiment_type: SentimentType
