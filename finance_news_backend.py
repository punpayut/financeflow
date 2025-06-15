# finance_news_backend.py

"""
FinanceFlow - AI Investment Assistant (Production Ready)
- Serves a modern frontend from /templates and /static folders.
- Aggregates news from multiple RSS feeds.
- Fetches real-time stock data.
- Uses AI for comprehensive news analysis, translation, and Q&A.
- Caches analyzed news in Firestore to save costs and improve speed.
"""

# --- Standard Library Imports ---
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

# --- Third-party Imports ---
import feedparser
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict

from groq import Groq
from flask import Flask, request, jsonify, render_template  # Use render_template
from flask_cors import CORS
import logging
from dotenv import load_dotenv

# Firebase Admin SDK for database
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
        logger.info(f"Initializing Firebase from file: {key_path}")
    else:
        cred = credentials.ApplicationDefault()
        logger.info("Initializing Firebase from Application Default Credentials.")

    firebase_admin.initialize_app(cred)
    db_firestore = firestore.client()
    analyzed_news_collection = db_firestore.collection('analyzed_news')
    logger.info("Firebase initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Failed to initialize Firebase: {e}. Caching will be disabled.", exc_info=True)
    analyzed_news_collection = None


# --- Constants ---
RSS_FEEDS = {
    'Yahoo Finance': 'https://finance.yahoo.com/news/rssindex',
    'Reuters Business': 'http://feeds.reuters.com/reuters/businessNews',
    'CNBC Top News': 'https://www.cnbc.com/id/100003114/device/rss/rss.html'
}

# --- Data Classes ---
@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    percent_change: float

@dataclass
class NewsItem:
    id: str # Use URL as ID
    title: str
    link: str
    source: str
    published: datetime
    content: str = ""
    analysis: Optional[Dict[str, Any]] = None

# --- Service Classes ---

class MarketDataProvider:
    """Fetches live stock market data."""
    def get_stock_data(self, symbols: List[str]) -> Dict[str, StockData]:
        if not symbols: return {}
        data = {}
        logger.info(f"Fetching stock data for: {symbols}")
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                info = tickers.tickers[symbol].fast_info
                price, prev_close = info.get('last_price'), info.get('previous_close')
                if price and prev_close:
                    change = price - prev_close
                    data[symbol] = StockData(
                        symbol=symbol,
                        price=round(price, 2),
                        change=round(change, 2),
                        percent_change=round((change / prev_close) * 100, 2)
                    )
            except Exception as e:
                logger.error(f"Could not fetch data for {symbol}: {e}")
        return data

class NewsAggregator:
    """Aggregates and scrapes news from multiple RSS feeds."""
    def _scrape_content(self, url: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'}
            response = requests.get(url, headers=headers, timeout=8)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'lxml')
            paragraphs = soup.find_all('p')
            text = ' '.join(p.get_text(strip=True) for p in paragraphs)
            return text[:2500]
        except Exception as e:
            logger.warning(f"Scraping failed for {url}: {e}")
            return ""

    def _fetch_from_feed(self, source_name: str, url: str) -> List[NewsItem]:
        logger.info(f"Fetching from {source_name}")
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:7]: # Get latest 7 from each source
            try:
                published_dt = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                items.append(NewsItem(id=entry.link, title=entry.title, link=entry.link, source=source_name, published=published_dt))
            except Exception as e:
                logger.warning(f"Could not parse entry from {source_name}: {e}")
        return items

    def get_latest_news(self, limit: int = 15) -> List[NewsItem]:
        all_items = []
        with ThreadPoolExecutor(max_workers=len(RSS_FEEDS)) as executor:
            future_to_feed = {executor.submit(self._fetch_from_feed, name, url): name for name, url in RSS_FEEDS.items()}
            for future in future_to_feed:
                all_items.extend(future.result())

        unique_items = {item.id: item for item in all_items}
        sorted_items = sorted(unique_items.values(), key=lambda x: x.published, reverse=True)
        
        top_items = sorted_items[:limit]
        with ThreadPoolExecutor(max_workers=5) as executor:
            scraped_contents = list(executor.map(self._scrape_content, [item.link for item in top_items]))
            for item, content in zip(top_items, scraped_contents):
                item.content = content
        
        return [item for item in top_items if len(item.content) > 150]

class AIProcessor:
    """Handles all interactions with the Groq LLM for advanced analysis."""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        if not self.client: logger.warning("GROQ_API_KEY not found.")
        self.model = "llama3-8b-8192"

    def analyze_news_item(self, news_item: NewsItem) -> Optional[Dict[str, Any]]:
        if not self.client or not news_item.content: return None
        prompt = f"""You are a top-tier financial analyst AI for an app called FinanceFlow. Analyze the provided news article.

        Source: {news_item.source}
        Title: {news_item.title}
        Content: {news_item.content}

        Return a JSON object with this exact structure:
        {{
          "summary_en": "A concise, one-paragraph summary of the article in English.",
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
        if not self.client: return "AI processor is offline."
        context_str = "\n\n".join([f"Title: {item.title}\nSummary: {item.analysis['summary_en']}" for item in news_context if item.analysis])
        prompt = f"""You are a helpful AI investment assistant. Answer the user's question in Thai based *only* on the provided context. Do not give direct financial advice. If the context is insufficient, state that.

        CONTEXT:
        ---
        {context_str}
        ---
        
        USER QUESTION: "{question}"

        YOUR ANSWER (in Thai):
        """
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
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Global cache for news to avoid re-fetching on every API call
news_cache: List[NewsItem] = []
cache_expiry = datetime.now()

def get_analyzed_news() -> List[NewsItem]:
    global news_cache, cache_expiry
    if not news_cache or datetime.now() > cache_expiry:
        logger.info("News cache expired. Fetching and analyzing...")
        fresh_news = news_aggregator.get_latest_news()
        
        # Check Firestore cache first
        items_to_analyze = []
        for item in fresh_news:
            if analyzed_news_collection:
                cached_doc = analyzed_news_collection.document(item.id).get()
                if cached_doc.exists:
                    item.analysis = cached_doc.to_dict()
                    logger.info(f"Firestore cache hit for {item.id}")
                else:
                    items_to_analyze.append(item)
            else:
                items_to_analyze.append(item)

        # Analyze only the items not found in cache
        if items_to_analyze:
            logger.info(f"Analyzing {len(items_to_analyze)} new articles.")
            with ThreadPoolExecutor(max_workers=5) as executor:
                analyses = list(executor.map(ai_processor.analyze_news_item, items_to_analyze))
            
            for item, analysis in zip(items_to_analyze, analyses):
                if analysis:
                    item.analysis = analysis
                    # Save new analysis to Firestore
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
        stock_data = market_provider.get_stock_data(list(all_symbols)[:10]) # Limit symbols for performance
        
        response_data = {
            "news": [asdict(item) for item in analyzed_news],
            "stocks": {symbol: asdict(data) for symbol, data in stock_data.items()}
        }
        return jsonify({"status": "success", "data": response_data})
    except Exception as e:
        logger.error(f"Error in main_feed endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Could not load feed."}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"status": "error", "message": "No question provided."}), 400

    news_context = news_cache # Use the already fetched news
    answer = ai_processor.answer_user_question(question, news_context)
    return jsonify({"status": "success", "answer": answer})

if __name__ == '__main__':
    print("🚀 Starting FinanceFlow [LOCAL DEVELOPMENT MODE]")
    app.run(debug=True, host='0.0.0.0', port=5000)