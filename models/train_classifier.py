import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump
from models.strategy_classifier import StrategyClassifier
import glob
from dotenv import load_dotenv
from news_module.news_fetcher import NewsFetcher
from news_module.sentiment_analyzer import NewsSentimentHandler
from datetime import datetime
from collections import defaultdict

load_dotenv()
api_key = os.getenv('NEWS_API_KEY')

# Load all XAUUSD data files from backtests
data_files = glob.glob('backtests/XAUUSD_*.csv')
if len(data_files) < 3:
    print("Warning: Less than 3 data files found. Expected M1, M5, M15.")
    exit(1)

tf_to_strategy = {
    'M1': 'scalping',
    'H4': 'swing',
    'M15': 'day_trading'
}

all_train_data = []

for file_path in data_files:
    # Extract timeframe from filename, e.g., XAUUSD_M1_20251104_214729.csv -> M1
    tf_str = file_path.split('/')[-1].split('_')[1]
    if tf_str not in tf_to_strategy:
        print(f"Skipping unknown timeframe: {tf_str}")
        continue
    strategy = tf_to_strategy[tf_str]

    # Load data
    df = pd.read_csv(file_path)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # Use relevant columns, rename tick_volume to volume
    df = df[['open', 'high', 'low', 'close', 'tick_volume']]
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)

    # Features (same as original)
    df['volatility'] = df['close'].rolling(window=10).std()
    df['momentum'] = df['close'].pct_change().rolling(5).mean()
    df['volume'] = df['volume']

    # Sentiment score from news
    if api_key:
        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date = df.index.max().strftime('%Y-%m-%d')
        print(f"Fetching news for {strategy} ({tf_str}) from {start_date} to {end_date}")
        fetcher = NewsFetcher(api_key, query="Gold OR XAUUSD")
        articles = fetcher.fetch_news(from_date=start_date, to_date=end_date)
        if articles:
            handler = NewsSentimentHandler()
            sentiment_results = handler.process_news(articles)
            # Group by date
            daily_sentiment = defaultdict(list)
            for res in sentiment_results:
                pub_date_str = res['publishedAt']
                pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00')).date()
                daily_sentiment[pub_date].append(res['score'])
            # Average per day
            daily_avg = {date: np.mean(scores) for date, scores in daily_sentiment.items()}
            # Assign to df
            df['sentiment_score'] = df.index.map(lambda t: daily_avg.get(t.date(), 0.0))
            print(f"Assigned sentiment for {len(daily_avg)} days for {strategy}")
        else:
            print(f"No news articles found for {strategy}, using random sentiment")
            df['sentiment_score'] = np.random.uniform(-1, 1, len(df))
    else:
        print("No API key, using random sentiment")
        df['sentiment_score'] = np.random.uniform(-1, 1, len(df))

    # Drop NaN
    df = df.dropna()

    # Add strategy label based on timeframe
    df['strategy'] = strategy

    # Append to list
    all_train_data.append(df[['volatility', 'volume', 'momentum', 'sentiment_score', 'strategy']])

# Combine all data
if not all_train_data:
    print("No data to train on.")
    exit(1)

train_df = pd.concat(all_train_data, ignore_index=True)

print(f"Combined data shape: {train_df.shape}")
print(f"Strategy distribution:\n{train_df['strategy'].value_counts()}")

# Train
classifier = StrategyClassifier()
classifier.train(train_df)

# Save
os.makedirs('models', exist_ok=True)
classifier.save_model('models/trained_classifier.joblib')
print('Model saved to models/trained_classifier.joblib')
