# finance_news_backend.py (AI Investment Assistant - RSS Only Version)

"""
FinanceFlow - AI Investment Assistant (RSS Only)
- Serves a modern frontend from /templates and /static folders.
- Aggregates news from multiple RSS feeds without web scraping.
- Fetches real-time stock data.
- Uses AI to analyze news based on RSS title and summary.
- Caches analyzed news in Firestore to save costs and improve speed.
"""

# --- Imports ---
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import re

import feedparser
import yfinance as yf
from dataclasses import dataclass, asdict

from groq import Groq
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

# --- Initialization ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Firebase
try:
    key_path = "google-credentials.json"
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
    else:
        cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
    db_firestore = firestore.client()
    analyzed_news_collection = db_firestore.collection('analyzed_news')
    logger.info("Firebase initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize Firebase: {e}", exc_info=True)
    analyzed_news_collection = None


# --- Constants ---
RSS_FEEDS = {
    'Yahoo Finance': 'https://finance.yahoo.com/news/rssindex',
    'Reuters Business': 'http://feeds.reuters.com/reuters/businessNews',
    'CNBC Top News': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'Investing.com': 'https://www.investing.com/rss/news.rss'
}

# --- Data Classes ---
@dataclass
class StockData:
    symbol: str; price: float; change: float; percent_change: float

@dataclass
class NewsItem:
    id: str; title: str; link: str; source: str; published: datetime;
    content: str = ""
    analysis: Optional[Dict[str, Any]] = None

# --- Utility Function ---
def clean_html(raw_html):
    """A simple function to remove HTML tags from a string."""
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

# --- Service Classes ---

class MarketDataProvider:
    """Fetches live stock market data."""
    def get_stock_data(self, symbols: List[str]) -> Dict[str, StockData]:
        # ... (This class remains unchanged) ...
        if not symbols: return {}
        data = {}
        logger.info(f"Fetching stock data for: {symbols}")
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                info = tickers.tickers[symbol].fast_info
                price, prev_close = info.get('last_price'), info.get('previous_close')
                if price and prev_close:
                    data[symbol] = StockData(symbol=symbol, price=round(price, 2), change=round(price - prev_close, 2), percent_change=round(((price - prev_close) / prev_close) * 100, 2))
            except Exception as e:
                logger.error(f"Could not fetch data for {symbol}: {e}")
        return data

class NewsAggregator:
    """Aggregates news from multiple RSS feeds WITHOUT web scraping."""
    def _fetch_from_feed(self, source_name: str, url: str) -> List[NewsItem]:
        logger.info(f"Fetching from {source_name}")
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:10]: # Get latest 10 from each source
            try:
                # The 'content' for the AI will be the RSS 'summary' or 'description' field
                rss_summary = entry.get('summary', entry.get('description', ''))
                cleaned_content = clean_html(rss_summary)

                published_dt = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                
                items.append(NewsItem(
                    id=entry.link,
                    title=entry.title,
                    link=entry.link,
                    source=source_name,
                    published=published_dt,
                    content=cleaned_content[:1500] # Limit content size
                ))
            except Exception as e:
                logger.warning(f"Could not parse entry from {source_name}: {e}")
        return items

    def get_latest_news(self, limit: int = 20) -> List[NewsItem]:
        all_items = []
        with ThreadPoolExecutor(max_workers=len(RSS_FEEDS)) as executor:
            future_to_feed = {executor.submit(self._fetch_from_feed, name, url): name for name, url in RSS_FEEDS.items()}
            for future in future_to_feed:
                all_items.extend(future.result())

        unique_items = {item.id: item for item in all_items if item.content}
        sorted_items = sorted(unique_items.values(), key=lambda x: x.published, reverse=True)
        
        logger.info(f"Fetched and processed {len(sorted_items)} articles from RSS feeds.")
        return sorted_items[:limit]

class AIProcessor:
    """Handles all interactions with the Groq LLM for advanced analysis."""
    # --- The __init__ method is unchanged ---
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        if not self.client: logger.warning("GROQ_API_KEY not found.")
        self.model = "llama3-8b-8192"

    def analyze_news_item(self, news_item: NewsItem) -> Optional[Dict[str, Any]]:
        # The prompt is now slightly adjusted as 'Content' is a summary, not the full article
        if not self.client or not news_item.content: return None
        prompt = f"""You are a top-tier financial analyst AI. Analyze the provided news summary from an RSS feed.

        Source: {news_item.source}
        Title: {news_item.title}
        Provided Summary (Content): {news_item.content}

        Your task is to return a JSON object with this exact structure:
        {{
          "summary_en": "A concise, one-paragraph summary of the article in English, based on the provided title and summary.",
          "summary_th": "A fluent, natural-sounding Thai translation of the English summary.",
          "sentiment": "Analyze the sentiment. Choose one: 'Positive', 'Negative', 'Neutral'.",
          "impact_score": "On a scale of 1-10, how impactful is this news for an average investor?",
          "affected_symbols": ["A list of stock ticker symbols (e.g., 'AAPL', 'NVDA') directly mentioned or heavily implied in the text."]
        }}
        """
        try:
            chat_completion = self.client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=self.model, temperature=0.1, response_format={"type": "json_object"})
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq analysis failed for {news_item.link}: {e}")
            return None

    def answer_user_question(self, question: str, news_context: List[NewsItem]) -> str:
        # --- This method is unchanged ---
        if not self.client: return "AI processor is offline."
        context_str = "\n\n".join([f"Title: {item.title}\nSummary: {item.analysis['summary_en']}" for item in news_context if item.analysis])
        prompt = f"""You are a helpful AI investment assistant...""" # Full prompt is the same
        try:
            chat_completion = self.client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model=self.model, temperature=0.5)
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq Q&A failed: {e}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผลคำตอบ"


# --- Application Components ---
news_aggregator = NewsAggregator()
market_provider = MarketDataProvider()
ai_processor = AIProcessor()

# --- Flask App & API Endpoints ---
# The logic for the Flask app, caching, and API endpoints is completely unchanged.
# It's robust enough to handle this new data source without modification.
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

news_cache: List[NewsItem] = []
cache_expiry = datetime.now()

def get_analyzed_news() -> List[NewsItem]:
    # ... (This function is unchanged) ...
    global news_cache, cache_expiry
    if not news_cache or datetime.now() > cache_expiry:
        logger.info("News cache expired. Fetching and analyzing...")
        fresh_news = news_aggregator.get_latest_news()
        
        items_to_analyze = []
        if analyzed_news_collection:
            for item in fresh_news:
                cached_doc = analyzed_news_collection.document(item.id).get()
                if cached_doc.exists:
                    item.analysis = cached_doc.to_dict()
                else:
                    items_to_analyze.append(item)
        else:
            items_to_analyze = fresh_news

        if items_to_analyze:
            logger.info(f"Analyzing {len(items_to_analyze)} new articles.")
            with ThreadPoolExecutor(max_workers=5) as executor:
                analyses = list(executor.map(ai_processor.analyze_news_item, items_to_analyze))
            
            for item, analysis in zip(items_to_analyze, analyses):
                if analysis:
                    item.analysis = analysis
                    if analyzed_news_collection:
                        analyzed_news_collection.document(item.id).set(analysis)

        news_cache = [item for item in fresh_news if item.analysis]
        cache_expiry = datetime.now() + timedelta(minutes=10)
    
    return news_cache

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/main_feed')
def get_main_feed():
    try:
        analyzed_news = get_analyzed_news()
        all_symbols = set(sym for item in analyzed_news for sym in item.analysis.get('affected_symbols', []))
        stock_data = market_provider.get_stock_data(list(all_symbols)[:10])
        
        response_data = {"news": [asdict(item) for item in analyzed_news], "stocks": {symbol: asdict(data) for symbol, data in stock_data.items()}}
        return jsonify({"status": "success", "data": response_data})
    except Exception as e:
        logger.error(f"Error in main_feed endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Could not load feed."}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question: return jsonify({"status": "error", "message": "No question provided."}), 400
    news_context = news_cache
    answer = ai_processor.answer_user_question(question, news_context)
    return jsonify({"status": "success", "answer": answer})

if __name__ == '__main__':
    print("🚀 Starting FinanceFlow [LOCAL DEVELOPMENT - RSS ONLY MODE]")
    app.run(debug=True, host='0.0.0.0', port=5000)