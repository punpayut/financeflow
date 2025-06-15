# finance_news_backend.py (Simplified Web App Version - Corrected Syntax)

"""
FinanceFlow Web App (Production Ready)
- This is the lightweight web server component of the FinanceFlow project.
- Its main responsibilities are:
  1. Serving the frontend application (HTML/CSS/JS).
  2. Fetching pre-analyzed news from Firestore.
  3. Fetching real-time stock prices.
  4. Handling user Q&A requests by querying the AI.
- It does NOT fetch or analyze RSS feeds; that is handled by worker.py.
"""

# --- Imports ---
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

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

# --- Firebase Initialization ---
try:
    key_path = "google-credentials.json"
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
    else:
        cred = credentials.ApplicationDefault()
    firebase_admin.initialize_app(cred)
    db_firestore = firestore.client()
    analyzed_news_collection = db_firestore.collection('analyzed_news')
    logger.info("WEB APP: Firebase initialized successfully.")
except Exception as e:
    logger.error(f"WEB APP: Failed to initialize Firebase: {e}", exc_info=True)
    analyzed_news_collection = None


# --- Data Classes ---
@dataclass
class StockData:
    symbol: str
    price: float
    change: float
    percent_change: float

@dataclass
class NewsItem:
    # This is used for context in the Q&A function
    id: str
    title: str
    link: str
    source: str
    published: any # Use 'any' to handle both datetime and string from Firestore
    content: str = ""
    analysis: Optional[Dict[str, Any]] = None

# --- Service Classes (Simplified for Web App) ---
class MarketDataProvider:
    """Fetches live stock market data."""
    def get_stock_data(self, symbols: List[str]) -> Dict[str, StockData]:
        if not symbols: return {}
        data = {}
        logger.info(f"WEB APP: Fetching stock data for: {symbols}")
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                info = tickers.tickers[symbol].fast_info
                price, prev_close = info.get('last_price'), info.get('previous_close')
                if price and prev_close:
                    data[symbol] = StockData(
                        symbol=symbol,
                        price=round(price, 2),
                        change=round(price - prev_close, 2),
                        percent_change=round(((price - prev_close) / prev_close) * 100, 2)
                    )
            except Exception as e:
                logger.error(f"WEB APP: Could not fetch stock data for {symbol}: {e}")
        return data

class AIProcessor:
    """This version of the processor is only for the Q&A feature."""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        if not self.client: logger.warning("WEB APP: GROQ_API_KEY not found for Q&A.")
        self.model = "llama3-8b-8192"

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
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.5
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"WEB APP: Groq Q&A failed: {e}")
            return "ขออภัยค่ะ เกิดข้อผิดพลาดในการประมวลผลคำตอบ"

# --- Application Components ---
market_provider = MarketDataProvider()
ai_processor_for_qa = AIProcessor()

# --- Flask App & API Endpoints ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/main_feed')
def get_main_feed():
    if not analyzed_news_collection:
        return jsonify({"status": "error", "message": "Database connection not available."}), 500
    try:
        # Query Firestore for the latest 25 news items, ordered by publish time
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(25)
        docs = query.stream()
        
        # Firestore returns documents, convert them to dictionaries
        news_from_db = [doc.to_dict() for doc in docs]
        
        # Extract all affected symbols from the news to fetch their prices
        all_symbols = set(sym.upper() for item in news_from_db for sym in item.get('analysis', {}).get('affected_symbols', []))
        
        # Fetch stock prices for the extracted symbols
        stock_data = market_provider.get_stock_data(list(all_symbols)[:20]) # Limit symbols for performance
        
        response_data = {
            "news": news_from_db,
            "stocks": {symbol: asdict(data) for symbol, data in stock_data.items()}
        }
        return jsonify({"status": "success", "data": response_data})
    except Exception as e:
        logger.error(f"WEB APP: Error fetching feed from Firestore: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Could not load feed from database."}), 500

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question')
    if not question:
        return jsonify({"status": "error", "message": "No question provided."}), 400
    
    if not analyzed_news_collection:
        return jsonify({"status": "error", "message": "Database connection not available for context."}), 500

    try:
        # Fetch latest news to use as context for the Q&A
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(10)
        docs = query.stream()
        
        # Re-create NewsItem objects from Firestore data for type consistency
        news_context = [NewsItem(**doc.to_dict()) for doc in docs]

        answer = ai_processor_for_qa.answer_user_question(question, news_context)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        logger.error(f"WEB APP: Error in Q&A endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Could not process question."}), 500

if __name__ == '__main__':
    print("🚀 Starting FinanceFlow Web App [LOCAL DEVELOPMENT MODE]")
    app.run(debug=True, host='0.0.0.0', port=5000)