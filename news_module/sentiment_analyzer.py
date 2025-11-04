from models.sentiment_model import SentimentAnalyzer
import logging

class NewsSentimentHandler:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.analyzer = SentimentAnalyzer(model_name)

    def process_news(self, articles):
        try:
            texts = [article["text"] for article in articles]
            results = self.analyzer.analyze_sentiment(texts)
            processed_results = []
            for i, result in enumerate(results):
                sentiment_score = 1 if result["sentiment"] == "bullish" else -1
                processed_results.append({
                    "publishedAt": articles[i]["publishedAt"],
                    "sentiment": result["sentiment"],
                    "score": result["score"] * sentiment_score
                })
            return processed_results
        except Exception as e:
            logging.error(f"Error processing sentiment: {e}")
            return []

if __name__ == '__main__':
    sample_articles = [
        {"text": "Gold rallies as inflation fears mount.", "publishedAt": "2025-11-04T10:00:00Z"},
        {"text": "XAU/USD drops amid Fed tightening expectations.", "publishedAt": "2025-11-04T12:00:00Z"}
    ]
    handler = NewsSentimentHandler()
    sentiment_results = handler.process_news(sample_articles)
    for item in sentiment_results:
        print(item)
