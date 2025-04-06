#!/usr/bin/env python3

from bertopic import BERTopic
from db import ReviewDB
import os

def train_and_save_bertopic_model(output_path="models/bertopic_model.pkl"):
    db = ReviewDB("reviews.db")
    reviews = db.get_all_reviews()
    db.close()

    documents = [r.body for r in reviews if r.body and r.body.strip()]
    print(f"📄 Training on {len(documents)} reviews...")

    topic_model = BERTopic()
    topic_model.fit(documents)

    # os.makedirs(output_path, exist_ok=True)
    topic_model.save(output_path)
    print(f"✅ Model saved to: {output_path}")

if __name__ == "__main__":
    train_and_save_bertopic_model()
