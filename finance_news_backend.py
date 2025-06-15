# finance_news_backend.py (Simplified Web App Version)

# --- Imports ---
import os
from datetime import datetime
from typing import List, Dict, Any

import yfinance as yf
from dataclasses import dataclass, asdict

from groq import Groq
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore

# --- Initialization (Unchanged) ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


# --- Data Classes (Unchanged) ---
@dataclass class StockData: ...
@dataclass class NewsItem: ...

# --- Service Classes (Simplified) ---
class MarketDataProvider:
    # ... (โค้ดเหมือนเดิม) ...
    
class AIProcessor:
    # --- The web app now only needs the Q&A part ---
    # ... (โค้ด __init__ และ answer_user_question เหมือนเดิม) ...


# --- Application Components ---
market_provider = MarketDataProvider()
ai_processor_for_qa = AIProcessor() # Instance for Q&A only

# --- Flask App & API Endpoints ---
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/main_feed')
def get_main_feed():
    """Fetches pre-analyzed news directly from Firestore."""
    if not analyzed_news_collection:
        return jsonify({"status": "error", "message": "Database connection not available."}), 500
    try:
        # Query Firestore for the latest 20 news items, ordered by publish time
        query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(20)
        docs = query.stream()
        
        news_from_db = [doc.to_dict() for doc in docs]
        
        # Convert timestamp objects back to datetime strings for JSON
        for news in news_from_db:
            if 'published' in news and isinstance(news['published'], datetime):
                news['published'] = news['published'].isoformat()
        
        all_symbols = set(sym.upper() for item in news_from_db for sym in item.get('analysis', {}).get('affected_symbols', []))
        stock_data = market_provider.get_stock_data(list(all_symbols)[:15])
        
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
    # ... (โค้ดเหมือนเดิม, แต่จะต้องดึง context จาก Firestore) ...
    data = request.get_json()
    question = data.get('question')
    if not question: return jsonify({"status": "error", "message": "No question provided."}), 400
    
    # Fetch latest news to use as context
    query = analyzed_news_collection.order_by("published", direction=firestore.Query.DESCENDING).limit(10)
    docs = query.stream()
    news_context = [NewsItem(**doc.to_dict()) for doc in docs]

    answer = ai_processor_for_qa.answer_user_question(question, news_context)
    return jsonify({"status": "success", "answer": answer})

if __name__ == '__main__':
    print("🚀 Starting FinanceFlow Web App [LOCAL DEVELOPMENT MODE]")
    app.run(debug=True, host='0.0.0.0', port=5000)