"""
Configuration settings for news module
"""
import os
from dotenv import load_dotenv

load_dotenv()

class NewsConfig:
    """Configuration class for news fetching"""
    
    # API Configuration
    NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')
    
    # Rate limiting settings
    MIN_REQUEST_INTERVAL = 1.0  # seconds between requests
    MAX_RETRIES = 3
    BASE_DELAY = 2  # seconds for exponential backoff
    
    # Cache settings
    CACHE_DURATION_HOURS = 1  # How long to keep cached news
    CACHE_FILE = "news_module/news_cache.json"
    
    # API request settings
    DEFAULT_PAGE_SIZE = 20  # Reduced from 100 to avoid rate limits
    MAX_PAGE_SIZE = 20  # Maximum allowed page size
    DEFAULT_LANGUAGE = "en"
    DEFAULT_QUERY = "Gold"
    
    # Fallback mode settings
    USE_FALLBACK_ON_FAILURE = True
    FALLBACK_SENTIMENT_SCORE = 0.0  # Neutral sentiment when no news available
    
    @classmethod
    def is_api_key_valid(cls) -> bool:
        """Check if the API key appears to be valid"""
        return bool(cls.NEWS_API_KEY and 
                   cls.NEWS_API_KEY != "your_newsapi_key_here" and
                   len(cls.NEWS_API_KEY) > 10)
    
    @classmethod
    def get_request_params(cls, query=None, language=None, page_size=None):
        """Get standard request parameters for NewsAPI"""
        return {
            "q": query or cls.DEFAULT_QUERY,
            "language": language or cls.DEFAULT_LANGUAGE,
            "pageSize": min(page_size or cls.DEFAULT_PAGE_SIZE, cls.MAX_PAGE_SIZE),
            "sortBy": "publishedAt",
            "apiKey": cls.NEWS_API_KEY
        }
    
    @classmethod
    def log_config_status(cls):
        """Log the current configuration status"""
        import logging
        
        if cls.is_api_key_valid():
            logging.info("NewsAPI configuration: Valid API key detected")
        else:
            logging.warning("NewsAPI configuration: Invalid or missing API key - fallback mode will be used")
        
        logging.info(f"News cache duration: {cls.CACHE_DURATION_HOURS} hours")
        logging.info(f"Max page size: {cls.MAX_PAGE_SIZE} articles per request")
        logging.info(f"Request interval: {cls.MIN_REQUEST_INTERVAL}s minimum")
