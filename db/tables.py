from dataclasses import dataclass
from typing import Optional

class Review:
    def __init__(self, id: Optional[int] = None, property_code: str = None, stars: float = None, title: str = None, date : str = None, body: str = None, age: str = None):
        self.id = id
        self.property_code = property_code
        self.stars = stars
        self.title = title
        self.date = date # Stored in 'YYYY-MM-DD' format
        self.body = body
        self.age = age

@dataclass
class ReviewTopic:
    review_id: int
    topic: str
    sentiment: str