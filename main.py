from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os
import pickle
import shutil

from tqdm import tqdm
from db.db import ReviewDB
from app.scraper import ChoiceReviewsScraper
from app.parser import ChoiceReviewsParser
from models.common import AugmentedReview, AugmentedTopic
from models.topic_sentiment import SentimentType, TopicSentimentAnalyzer
from db.tables import Review, ReviewTopic




def main():
    property_code = "ny900"
    db = ReviewDB("reviews.db")
    # db.init_db()

    # url = f"https://www.choicehotels.com/{property_code}"
    # scraper = ChoiceReviewsScraper(url)

    # review_blocks, driver = scraper.get_all_review_blocks(url)

    # for idx, block in enumerate(review_blocks, 1):
    #     try:
    #         parser = ChoiceReviewsParser()
    #         review = parser.parse_choice_review_block(block, property_code)
    #         review.id = db.insert_review(review)
    #     except Exception as e:
    #         print(f"⚠️ Error on review {idx}: {e}")

    # print(f"✅ Inserted {len(review_blocks)} reviews")
    # driver.quit()

    # ✅ Run topic + sentiment analysis for all reviews for this property
    analyzer = TopicSentimentAnalyzer()
    reviews: list[Review] = db.get_all_reviews()
    reviews_text: list[str] = db.get_reviews_as_list()
    db.close()

    augmented_topics = []
    augmented_reviews = []


    # Extract topics from reviews
    analyzer = TopicSentimentAnalyzer()
    topics = analyzer.analyze_text(reviews_text)

    # Analyze topic sentiment
    for t in topics:
        augmented_topics.append(AugmentedTopic(t, analyzer.analyze_sentiment_topic(t)))

    # Augment topics for UI
    for r in tqdm(reviews):
        assigned_topics = analyzer.assign_topics_via_chatgpt(r.body, topics)
        augmented_reviews.append(AugmentedReview(r, assigned_topics, analyzer.analyze_sentiment_review(r.body), datetime.strptime(r.date, "%Y-%m-%d")))


    # Remove previouslt saved data
    if os.path.exists(f'data/{property_code}'):
        shutil.rmtree(f'data/{property_code}')
    os.mkdir(f'data/{property_code}')

    # Save processed parameters
    with open(f'data/{property_code}/reviews.pkl', 'wb') as f:
        pickle.dump(augmented_reviews, f)
    with open(f'data/{property_code}/topics.pkl', 'wb') as f:
        pickle.dump(augmented_topics, f)


if __name__ == "__main__":
    main()
