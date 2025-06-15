# finance_news_backend.py (Data Sanitization Fix Version)

"""
FinanceFlow Web App (Production Ready)
- This is the lightweight web server component of the FinanceFlow project.
- Its main responsibilities are:
  1. Serving the frontend application (HTML/CSS/JS).
  2. Fetching pre-analyzed news from Firestore.
  3. Fetching real-time stock prices.
  4. Handling user Q&A requests by querying the AI.
- Includes a data sanitization step to validate ticker symbols from the AI.
"""

# --- Imports ---
import os
import re  # <-- Import for regular expressions
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
    published: Any # Use 'any' to handle both datetime and string from Firestore
    content: str = ""
    analysis: Optional[Dict[str, Any]] = None

# --- NEW Utility Function ---
def is_valid_ticker(symbol: str) -> bool:
    """A simple validator to filter out invalid ticker symbols returned by the AI."""
    if not symbol or not isinstance(symbol, str):
        return False  # Filter out empty or non-string symbols
    if len(symbol) > 6 or len(symbol) < 1:
        return False  # Tickers are typically 1-6 characters
    if ' ' in symbol:
        return False  # Tickers do not contain spaces
    # A valid ticker contains uppercase letters, and may contain a dot or a dash.
    if not re.match(r'^[A-Z0-9.-]+$', symbol):
        return False
    return True

# --- Service Classes ---
class MarketDataProvider:
    """Fetches live stock market data."""
    def get_stock_data(self, symbols: List[str]) -> Dict[str, StockData]:
        if not symbols: return {}
        data = {}
        logger.info(f"WEB APP: Fetching stock data for: {symbols}")
        try:
            tickers = yf.Tickers(' '.join(symbols))
            for symbol in symbols:
                try:
                    # Use .info which is more comprehensive, or stick to fast_info
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
                    # Log error for a specific ticker but continue with others
                    logger.warning(f"WEB APP: Could not fetch data for an individual ticker '{symbol}': {e}")
        except Exception as e:
            logger.error(f"WEB APP: A general error occurred in yfinance Tickers call: {e}")
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
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(25)
        docs = query.stream()
        news_from_db = [doc.to_dict() for doc in docs]
        
        # --- DATA SANITIZATION LOGIC ---
        
        # 1. Gather all raw symbols suggested by the AI
        raw_symbols = set(sym.upper() for item in news_from_db for sym in item.get('analysis', {}).get('affected_symbols', []))
        
        # 2. Filter for valid ticker formats
        valid_symbols = {sym for sym in raw_symbols if is_valid_ticker(sym)}
        
        logger.info(f"WEB APP: Raw symbols from AI: {raw_symbols}")
        logger.info(f"WEB APP: Validated symbols for fetching: {valid_symbols}")
        
        # 3. Fetch stock prices using only the cleaned list of symbols
        stock_data = market_provider.get_stock_data(list(valid_symbols)[:20])
        
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
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(10)
        docs = query.stream()
        news_context = [NewsItem(**doc.to_dict()) for doc in docs]

        answer = ai_processor_for_qa.answer_user_question(question, news_context)
        return jsonify({"status": "success", "answer": answer})
    except Exception as e:
        logger.error(f"WEB APP: Error in Q&A endpoint: {e}", exc_info=True)
        return jsonify({"status": "error", "message": "Could not process question."}), 500

if __name__ == '__main__':
    print("🚀 Starting FinanceFlow Web App [LOCAL DEVELOPMENT MODE]")
    app.run(debug=True, host='0.0.0.0', port=5000)