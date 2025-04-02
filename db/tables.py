from dataclasses import dataclass
from typing import Optional

@dataclass
class Review:
    property_code: str
    stars: float
    title: str
    date: str  # Stored in 'YYYY-MM-DD' format
    body: str
    age: str = None
    id: Optional[int] = None

@dataclass
class ReviewTopic:
    review_id: int
    topic: str
    sentiment: str