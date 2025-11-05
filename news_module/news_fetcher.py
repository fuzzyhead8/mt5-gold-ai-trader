import requests
import logging
import time
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional

class NewsFetcher:
    def __init__(self, api_key, query="Gold", language="en", cache_duration_hours=1):
        self.api_key = api_key
        self.query = query
        self.language = language
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache_file = "news_module/news_cache.json"
        self.cache_duration = timedelta(hours=cache_duration_hours)
        self.last_request_time = 0
        self.min_request_interval = 120.0  # 2 minutes between requests - much more conservative
        self.max_retries = 2  # Reduced retries
        self.base_delay = 300  # 5 minute base delay for exponential backoff
        self.daily_request_count = 0
        self.daily_limit = 50  # Conservative daily limit (NewsAPI free tier allows 100/day)
        self.last_reset_date = datetime.now().date()
        
        # Fallback news data for when API fails
        self.fallback_news = [
            {
                "text": "Gold prices show stability amid market uncertainty. Investors continue to monitor economic indicators.",
                "publishedAt": datetime.now().isoformat()
            },
            {
                "text": "Central bank policies impact gold trading volumes. Market sentiment remains cautious.",
                "publishedAt": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]

    def _load_cache(self) -> Optional[Dict]:
        """Load cached news data if it exists and is still valid"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', ''))
                if datetime.now() - cache_time < self.cache_duration:
                    logging.info("Using cached news data")
                    return cache_data.get('articles', [])
        except Exception as e:
            logging.warning(f"Failed to load news cache: {e}")
        return None

    def _save_cache(self, articles: List[Dict]) -> None:
        """Save news data to cache"""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'articles': articles
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logging.info("News data cached successfully")
        except Exception as e:
            logging.warning(f"Failed to save news cache: {e}")

    def _rate_limit(self) -> None:
        """Implement conservative rate limiting between requests"""
        # Reset daily counter if it's a new day
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = current_date
            
        # Check daily limit
        if self.daily_request_count >= self.daily_limit:
            logging.warning(f"Daily API request limit ({self.daily_limit}) reached. Using cache/fallback only.")
            raise Exception("Daily API limit exceeded")
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            logging.info(f"Rate limiting: sleeping for {sleep_time:.0f} seconds")
            time.sleep(sleep_time)
        self.last_request_time = time.time()
        self.daily_request_count += 1

    def _make_request_with_retry(self, params: Dict) -> Optional[List[Dict]]:
        """Make API request with exponential backoff retry logic"""
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                
                response = requests.get(self.base_url, params=params, timeout=10)
                
                if response.status_code == 200:
                    articles = response.json().get("articles", [])
                    processed_articles = []
                    for article in articles:
                        if article.get("description"):
                            text = article["title"] + ". " + article["description"]
                            published_at = article["publishedAt"]
                            processed_articles.append({
                                "text": text,
                                "publishedAt": published_at
                            })
                    return processed_articles
                
                elif response.status_code == 429:  # Too Many Requests
                    retry_after = response.headers.get('Retry-After')
                    if retry_after:
                        wait_time = max(int(retry_after), 600)  # At least 10 minutes
                    else:
                        wait_time = self.base_delay * (2 ** attempt)  # Starts at 5 minutes, then 10, then 20
                    
                    logging.error(f"Rate limited (429). Waiting {wait_time} seconds before retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                    continue
                
                elif response.status_code == 426:  # Upgrade Required (API key issue)
                    logging.error("API key issue or subscription limit reached")
                    break
                
                else:
                    response.raise_for_status()
                    
            except requests.exceptions.RequestException as e:
                logging.warning(f"Request attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.base_delay * (2 ** attempt)
                    logging.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                continue
            except Exception as e:
                logging.error(f"Unexpected error in news fetch attempt {attempt + 1}: {e}")
                break
        
        return None

    def fetch_news(self, from_date=None, to_date=None, page_size=10) -> List[Dict]:
        """
        Fetch news with very conservative error handling, caching, and rate limiting
        Much reduced default page_size to minimize API usage
        """
        # Always try to load from cache first - even if expired
        cached_articles = self._load_cache()
        if cached_articles:
            logging.info("Using cached news data to avoid API calls")
            return cached_articles

        # Check daily limit before attempting API call
        current_date = datetime.now().date()
        if current_date > self.last_reset_date:
            self.daily_request_count = 0
            self.last_reset_date = current_date
            
        if self.daily_request_count >= self.daily_limit:
            logging.warning(f"Daily API request limit ({self.daily_limit}) reached. Using fallback data.")
            return self.fallback_news

        # If no valid API key, return fallback
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            logging.warning("No valid API key provided. Using fallback news data.")
            return self.fallback_news

        # Very conservative API parameters
        params = {
            "q": self.query,
            "language": self.language,
            "pageSize": min(page_size, 10),  # Much smaller limit
            "sortBy": "publishedAt",
            "apiKey": self.api_key
        }
        
        if from_date:
            params["from"] = from_date
        if to_date:
            params["to"] = to_date

        # Try to fetch from API with conservative approach
        try:
            articles = self._make_request_with_retry(params)
            
            if articles:
                self._save_cache(articles)
                logging.info(f"Successfully fetched {len(articles)} news articles (API calls today: {self.daily_request_count})")
                return articles
            else:
                logging.warning("API attempts failed. Using fallback news data.")
                return self.fallback_news
        except Exception as e:
            logging.warning(f"News fetch exception: {e}. Using fallback data.")
            return self.fallback_news

    def get_cached_or_fallback(self) -> List[Dict]:
        """Get cached news or fallback data without making API calls"""
        cached_articles = self._load_cache()
        if cached_articles:
            return cached_articles
        return self.fallback_news

if __name__ == '__main__':
    # Replace with your real API key
    API_KEY = "your_newsapi_key_here"
    fetcher = NewsFetcher(api_key=API_KEY)
    articles = fetcher.fetch_news()
    for i, article in enumerate(articles, 1):
        print(f"{i}. {article['text']} - {article['publishedAt']}\n")
