from db.db import ReviewDB
from app.scraper import ChoiceReviewsScraper
from app.parser import ChoiceReviewsParser

def main():
    db = ReviewDB("reviews.db")
    db.init_db()

    property_code = "ny900"
    url = "https://www.choicehotels.com/"+property_code
    choice_reviews_scraper = ChoiceReviewsScraper(url)

    review_blocks, driver= choice_reviews_scraper.get_all_review_blocks(url)

    for idx, block in enumerate(review_blocks, 1):
        try:
            parser = ChoiceReviewsParser()
            review = parser.parse_choice_review_block(block, property_code)
            review.id = db.insert_review(review)
        except Exception as e:
            print(f"⚠️ Error on review {idx}: {e}")

    print(f"✅ Inserted {len(review_blocks)} reviews")
    driver.quit()
    db.close()


if __name__ == "__main__":
    main()
