#!/usr/bin/env python3

from db.db import ReviewDB
from models.topic_sentiment import TopicSentimentAnalyzer
from db.tables import ReviewTopic

analyzer = TopicSentimentAnalyzer()
db = ReviewDB("reviews.db")

# Assuming your DB class has a method like get_all_reviews() returning [(id, body), ...]
reviews = db.get_all_reviews()

for review_id, body in reviews:
    if body:
        topic_entries: list[ReviewTopic] = analyzer.analyze_text(review_id, body)
        db.insert_review_topics(topic_entries)

db.close()
