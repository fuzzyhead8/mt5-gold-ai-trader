import requests
import logging

class NewsFetcher:
    def __init__(self, api_key, query="Gold", language="en"):
        self.api_key = api_key
        self.query = query
        self.language = language
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, from_date=None, to_date=None, page_size=100):
        params = {
            "q": self.query,
            "language": self.language,
            "pageSize": page_size,
            "sortBy": "publishedAt",
            "apiKey": self.api_key
        }
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            processed_articles = []
            for article in articles:
                if article["description"]:
                    text = article["title"] + ". " + article["description"]
                    published_at = article["publishedAt"]
                    processed_articles.append({
                        "text": text,
                        "publishedAt": published_at
                    })
            return processed_articles
        except Exception as e:
            logging.error(f"Failed to fetch news: {e}")
            return []

if __name__ == '__main__':
    # Replace with your real API key
    API_KEY = "your_newsapi_key_here"
    fetcher = NewsFetcher(api_key=API_KEY)
    articles = fetcher.fetch_news()
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['text']} - {article['publishedAt']}\n")
