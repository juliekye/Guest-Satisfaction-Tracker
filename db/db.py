import sqlite3
from db.tables import Review, ReviewTopic


class ReviewDB:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

    def close(self):
        self.conn.close()

    def init_db(self):
        """Creates tables if they don't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                property_code TEXT,
                stars FLOAT,
                title TEXT,
                date DATE,
                body TEXT,
                age TEXT
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_id INTEGER,
                topic TEXT,
                sentiment TEXT,
                FOREIGN KEY(review_id) REFERENCES reviews(id)
            )
        """)
        self.conn.commit()

    def insert_review(self, review: Review) -> int:
        self.cursor.execute("""
            INSERT INTO reviews (property_code, stars, title, date, body, age)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (review.property_code, review.stars, review.title, review.date, review.body, review.age))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_reviews_by_property_code(self, property_code: str) -> list[Review]:
        """Returns all Review objects for a specific property_code with non-empty body."""
        self.cursor.execute("""
            SELECT id, property_code, stars, title, date, body, age
            FROM reviews
            WHERE property_code = ?
            AND body IS NOT NULL
            AND TRIM(body) != ''
        """, (property_code,))
        rows = self.cursor.fetchall()
        return [Review(*row) for row in rows]

    
    def get_all_reviews(self) -> list[Review]:
        self.cursor.execute("""
            SELECT id, property_code, stars, title, date, body, age
            FROM reviews
            WHERE body IS NOT NULL
            AND TRIM(body) != ''
        """)
        rows = self.cursor.fetchall()
        return [Review(*row) for row in rows]


    def insert_review_topics(self, topic_entries: list[ReviewTopic]):
        """Bulk insert of ReviewTopic objects into review_topics."""
        self.cursor.executemany("""
            INSERT INTO review_topics (review_id, topic, sentiment)
            VALUES (?, ?, ?)
        """, [(t.review_id, t.topic, t.sentiment) for t in topic_entries])
        self.conn.commit()

#   TO DO**************
  #WRONG REVIEW ID 