from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from bertopic import BERTopic

class TopicSentimentAnalyzer:
    def __init__(self, model_path="bertopic_model"):
        self.topic_model = BERTopic.load(model_path)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, review_id, text):
        results = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            topic_id = self.topic_model.transform([sentence])[0][0]
            topic_words = self.topic_model.get_topic(topic_id)
            topic_label = ', '.join([word for word, _ in topic_words[:3]]) if topic_words else 'unknown'

            score = self.sentiment_analyzer.polarity_scores(sentence)['compound']
            sentiment = (
                'positive' if score > 0.05 else
                'negative' if score < -0.05 else
                'neutral'
            )

            results.append((review_id, topic_label, sentiment))
        return results
