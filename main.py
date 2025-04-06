from db.db import ReviewDB
# from app.scraper import ChoiceReviewsScraper
# from app.parser import ChoiceReviewsParser
from models.topic_sentiment import TopicSentimentAnalyzer
from db.tables import ReviewTopic

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
    reviews = db.get_reviews_by_property_code(property_code)

    for review in reviews:
        if review.body:
            topics: list[ReviewTopic] = analyzer.analyze_text(review.id, review.body)
            db.insert_review_topics(topics)

    print("✅ Topic + sentiment analysis complete.")
    db.close()

if __name__ == "__main__":
    main()
