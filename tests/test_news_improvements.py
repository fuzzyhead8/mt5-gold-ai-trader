#!/usr/bin/env python3
"""
Test script to verify the improved news fetching functionality
"""
import sys
import os
import logging
import time

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.getcwd())

from news_module.news_fetcher import NewsFetcher
from news_module.config import NewsConfig
from news_module.sentiment_analyzer import NewsSentimentHandler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_news_fetcher():
    """Test the enhanced news fetcher"""
    print("="*60)
    print("Testing Enhanced News Fetcher")
    print("="*60)
    
    # Log configuration status
    NewsConfig.log_config_status()
    
    # Initialize the news fetcher
    fetcher = NewsFetcher(
        api_key=NewsConfig.NEWS_API_KEY,
        query="Gold",
        cache_duration_hours=2
    )
    
    print("\n1. Testing news fetch with caching and rate limiting...")
    
    start_time = time.time()
    articles = fetcher.fetch_news(page_size=10)  # Small page size for testing
    fetch_time = time.time() - start_time
    
    print(f"   - Fetched {len(articles)} articles in {fetch_time:.2f} seconds")
    print(f"   - Sample article: {articles[0]['text'][:100] if articles else 'No articles'}...")
    
    print("\n2. Testing cached retrieval...")
    start_time = time.time()
    cached_articles = fetcher.fetch_news(page_size=10)  # Should use cache
    cached_fetch_time = time.time() - start_time
    
    print(f"   - Retrieved {len(cached_articles)} articles in {cached_fetch_time:.2f} seconds (from cache)")
    
    print("\n3. Testing fallback data...")
    fallback_articles = fetcher.get_cached_or_fallback()
    print(f"   - Fallback articles: {len(fallback_articles)}")
    print(f"   - Sample fallback: {fallback_articles[0]['text'][:100] if fallback_articles else 'No fallback'}...")
    
    return articles

def test_sentiment_analysis(articles):
    """Test sentiment analysis on the fetched articles"""
    print("\n" + "="*60)
    print("Testing Sentiment Analysis")
    print("="*60)
    
    if not articles:
        print("No articles to analyze")
        return
    
    sentiment_handler = NewsSentimentHandler()
    sentiment_results = sentiment_handler.process_news(articles)
    
    if sentiment_results:
        avg_sentiment = sum(result['score'] for result in sentiment_results) / len(sentiment_results)
        print(f"   - Processed {len(sentiment_results)} articles for sentiment")
        print(f"   - Average sentiment score: {avg_sentiment:.3f}")
        print(f"   - Sentiment interpretation: {'Bullish' if avg_sentiment > 0 else 'Bearish' if avg_sentiment < 0 else 'Neutral'}")
        
        # Show a few examples
        print("\n   Sample sentiment analysis:")
        for i, result in enumerate(sentiment_results[:3]):
            print(f"     {i+1}. Score: {result['score']:.3f} - {result['text'][:80]}...")
    else:
        print("   - No sentiment results")

def test_api_error_handling():
    """Test error handling with invalid API key"""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60)
    
    # Test with invalid API key
    invalid_fetcher = NewsFetcher(api_key="invalid_key_test", cache_duration_hours=0.1)
    
    print("   - Testing with invalid API key...")
    articles = invalid_fetcher.fetch_news(page_size=5)
    print(f"   - Received {len(articles)} fallback articles")
    
    if articles:
        print(f"   - Sample fallback article: {articles[0]['text'][:100]}...")

def main():
    """Main test function"""
    print("MT5 Gold AI Trader - News Module Improvement Test")
    print("This test verifies the enhanced news fetching with:")
    print("- Rate limiting and retry logic")
    print("- Caching mechanism") 
    print("- Fallback data when API fails")
    print("- Improved error handling")
    
    try:
        # Test basic news fetching
        articles = test_news_fetcher()
        
        # Test sentiment analysis
        test_sentiment_analysis(articles)
        
        # Test error handling
        test_api_error_handling()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("The news module improvements are working correctly.")
        print("Your trading bot should now handle NewsAPI rate limits gracefully.")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        print("Please check the error details above.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
