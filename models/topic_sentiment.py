from enum import Enum
import json
import pandas as pd
import tensorflow as tf
import openai

import umap
import hdbscan
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import urllib
from models.common import SentimentType
from transformers import pipeline
from sentence_transformers import CrossEncoder


class TopicSentimentAnalyzer:
    def __init__(self):
        """
        In the constructor, we set up our custom HDBSCAN, custom UMAP,
        and create a BERTopic model that uses them. 
        You can adjust these parameters to match your data domain.
        """
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=2,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        self.umap_model = umap.UMAP(
            n_neighbors=3,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            low_memory=False
        )

        self.vectorizer_model = CountVectorizer(
            stop_words='english',
            ngram_range=(2, 3),
            min_df=2
        )

        self.topic_model = BERTopic(
            calculate_probabilities=True,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model
        )

        self.review_sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True
        )

        self.topic_sentiment_classifier = pipeline(
            "sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            truncation=True
        )

        self.sentence_transformer = SentenceTransformer("all-mpnet-base-v2")

        self.sentiment_tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment')

        # Load labels from mapping file
        mapping_url = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/mapping.txt"
        with urllib.request.urlopen(mapping_url) as f:
            lines = f.read().decode("utf-8").splitlines()
        self.sentiment_labels = [row.split('\t')[1] for row in lines if row]

    def preprocess_text(self, text: str) -> str:
        """
        This replicates the preprocessing logic from the article using TF ops.
        If you prefer, you can replace this with a pure Python/re-based approach.
        """
        # Convert text to a tensor for tf.strings operations
        tensor_text = tf.convert_to_tensor(text)
        
        # Convert to lowercase
        tensor_text = tf.strings.lower(tensor_text)
        
        # Remove HTML tags
        tensor_text = tf.strings.regex_replace(tensor_text, r'<.*?>', '')
        
        # Remove non-alphabet characters
        tensor_text = tf.strings.regex_replace(tensor_text, r'[^a-zA-Z\s]', '')
        
        # Strip leading and trailing whitespaces
        tensor_text = tf.strings.strip(tensor_text)
        
        # Convert back to normal Python string
        return tensor_text.numpy().decode('utf-8')

    def analyze_text(self, reviews: list[str]):
        """
        This method:
        1) Creates a DataFrame from the list of reviews.
        2) Drops NaN reviews, removes reviews shorter than 20 chars.
        3) Applies text-cleaning via preprocess_text().
        4) Fits BERTopic to the cleaned reviews.
        5) Optionally reduces the number of topics to 100
        6) Updates topics (increasing n-gram range to (1, 3)) for better representation.
        
        Returns:
        topics: A list indicating the topic index assigned to each document.
        probs: The topic probability distribution per document.
        topic_keywords: Dictionary mapping each topic ID (int) to a list of top terms (str).
        """
        # Create a DataFrame from the reviews
        df = pd.DataFrame({'review_description': reviews})

        # Drop rows with NaN values in the 'review_description'
        df.dropna(subset=['review_description'], inplace=True)

        # Remove rows where the length of review_description < 20
        df = df[df['review_description'].str.len() >= 20]

        # Preprocess each review
        df['cleaned_review'] = df['review_description'].apply(self.preprocess_text)

        # Fit the model on cleaned reviews
        topics, probs = self.topic_model.fit_transform(df['cleaned_review'].tolist())

        # Reduce the number of topics to 100 (optional)
        # self.topic_model.reduce_topics(df['cleaned_review'].tolist(), nr_topics=100)

        # Update topic representations (expand n-gram range to 1–3)
        self.topic_model.update_topics(
            df['cleaned_review'].tolist(), 
            topics=self.topic_model.topics_,
            n_gram_range=(2, 3)
        )

        # Build a dictionary mapping topic IDs to the top words (exclude -1 outliers)
        topic_info = self.topic_model.get_topic_info()
        topic_keywords = {}
        for row in topic_info.itertuples():
            if row.Topic != -1:  # Skip the outlier cluster labeled as -1
                top_terms = self.topic_model.get_topic(row.Topic)
                # top_terms is a list of (word, relevance_score) pairs
                topic_keywords[row.Topic] = [term for term, _ in top_terms]

        topic_sentences = self.generate_consolidated_topic_labels(topic_keywords)
        print('Generated topics: ', topic_sentences)
        return topic_sentences

    def assign_topics_via_chatgpt(self, review: str, available_topics: list[str]) -> list[str]:
        """
        Uses GPT to assign relevant topics from available list to a given review.
        Returns a subset of available_topics that apply to the review.
        """
        # Load API key from secrets.json
        with open("secrets.json", "r") as f:
            secrets = json.load(f)
        client = openai.OpenAI(api_key=secrets["CHAT_GPT_API_KEY"])

        topic_list_str = "\n".join(f"- {t}" for t in available_topics)
        prompt = (
            f"You are an assistant that reads hotel reviews and assigns relevant topics.\n\n"
            f"Available topics:\n{topic_list_str}\n\n"
            f"Review:\n\"{review}\"\n\n"
            f"Which of the above topics are mentioned or implied in this review?\n"
            f"Return only a Python list of topic strings, no explanations. Do not use ```python\n - return raw text"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant that assigns relevant topics to hotel reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.0
        )

        # Parse output as Python list
        import ast
        try:
            q = response.choices[0].message.content.strip()
            assigned = ast.literal_eval(response.choices[0].message.content.strip())
            if isinstance(assigned, list):
                return assigned
            return []
        except Exception:
            return []
    
    def generate_consolidated_topic_labels(self, topic_keywords: dict[int, list[str]]) -> list[str]:
        # Load API key from secrets.json
        with open("secrets.json", "r") as f:
            secrets = json.load(f)
        client = openai.OpenAI(api_key=secrets["CHAT_GPT_API_KEY"])

        # Prepare structured input
        topic_blocks = []
        for topic_id, keywords in topic_keywords.items():
            kw_line = ", ".join(keywords)
            topic_blocks.append(f"Topic {topic_id}: {kw_line}")
        all_topics_text = "\n".join(topic_blocks)

        # print(topic_keywords)

        # Construct a single prompt
        prompt = (
            f"You are given keywords for multiple topics:\n\n{all_topics_text}\n\n"
            "For each topic, extract 1 to 3 short and useful subtopics (max 3-4 words each).\n"
            "Each subtopic must reflect a purpose and sentiment (positive or negative).\n"
            "Avoid repeating subtopics across topics. Avoid vague terms. Use lowercase only.\n"
            "Return a flat, numbered list with no extra commentary.\n"
            "Do not generate similar very topics! It should be clear what each topic means just by looking at it! If some topics have somewhat similar subtopics - do not repeat them, each item you generate must be uniquie - it is really important!"
            "Example:\n"
            "1. clean room condition\n"
            "2. rude front desk staff\n"
            "3. great hotel location"
            "Examples of bad subtopics that are too similar: "
        )

        # Call ChatGPT
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You extract concise and useful subtopics with sentiment."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.3
        )

        # Extract and clean the output
        raw_lines = response.choices[0].message.content.strip().splitlines()
        labels = [line.split(". ", 1)[-1].strip() for line in raw_lines if ". " in line]

        # Remove duplicates while preserving order
        seen = set()
        unique_labels = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                unique_labels.append(label)

        return unique_labels


    def analyze_sentiment_topic(self, topic: str) -> SentimentType:
        """
        Analyzes sentiment of a short topic label using Siebert's RoBERTa (binary).
        """
        result = self.topic_sentiment_classifier(topic)[0]
        label = result['label'].strip().lower()

        if label == "positive":
            return SentimentType.POSITIVE
        elif label == "negative":
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL

    def analyze_sentiment_review(self, review: str) -> SentimentType:
        """
        Analyzes sentiment of a full review using DistilBERT SST-2 (binary).
        """
        result = self.review_sentiment_classifier(review[:512])[0]
        label = result['label'].strip().lower()

        if label == "positive":
            return SentimentType.POSITIVE
        elif label == "negative":
            return SentimentType.NEGATIVE
        else:
            return SentimentType.NEUTRAL
