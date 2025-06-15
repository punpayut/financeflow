# finance_news_backend.py

"""
FinanceFlow - Production Ready for Render.com with Firebase Firestore
- Uses .env for local development via python-dotenv
- Correctly initializes Firebase on Render by reading the Secret File from its path
- Uses Firestore for persistent, free database caching
- Uses Gunicorn as the production WSGI server
"""

# --- Standard Library Imports ---
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# --- Third-party Imports ---
import requests
from dataclasses import dataclass, asdict
from groq import Groq
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
from dotenv import load_dotenv

# Firebase Admin SDK for database
import firebase_admin
from firebase_admin import credentials, firestore

# --- Load .env and Initialize Services ---
# This line loads variables from the .env file into the environment.
# It's crucial for local development.
load_dotenv()

# Configure logging to see output in Render's logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Firebase Initialization (Robust version for Render Secret Files) ---
try:
    # Define the path for the credentials file.
    # On Render, the Secret File is placed in the root directory of the application.
    key_path = "google-credentials.json"

    # Check if the credentials file exists at the expected path.
    if os.path.exists(key_path):
        # This block will run on Render
        logger.info(f"Found credentials file at: {key_path}. Initializing Firebase from file.")
        # Initialize using the service account file path directly.
        cred = credentials.Certificate(key_path)
    else:
        # This block will run on local development (or if the file is missing on Render)
        logger.info(f"'{key_path}' not found. Falling back to GOOGLE_APPLICATION_CREDENTIALS env var.")
        # This relies on the GOOGLE_APPLICATION_CREDENTIALS environment variable
        # set in the .env file for local development.
        cred = credentials.ApplicationDefault()

    # Initialize the Firebase app with the determined credentials
    firebase_admin.initialize_app(cred)
    logger.info("Firebase initialized successfully.")
    
    # Get a client to the Firestore service
    db_firestore = firestore.client()
    # Create a reference to the collection we'll use for caching summaries
    summaries_collection = db_firestore.collection('summaries')

except Exception as e:
    logger.error(f"FATAL: Failed to initialize Firebase: {e}. Caching will be disabled.", exc_info=True)
    summaries_collection = None


# --- Data Classes (Application's internal data structures) ---
@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    category: str = "general"

@dataclass
class Summary:
    article_id: str
    simple_summary: str
    key_points: List[str]
    impact_analysis: str
    investment_implications: str
    difficulty_level: str
    reading_time_minutes: int


# --- FirestoreManager (Handles all database operations) ---
class FirestoreManager:
    """A manager class to abstract Firestore operations."""
    
    def find_summary_by_article_id(self, article_id: str) -> Optional[Summary]:
        if not summaries_collection: return None
        try:
            doc = summaries_collection.document(article_id).get()
            if doc.exists:
                logger.info(f"Cache hit in Firestore for article ID {article_id}.")
                return Summary(**doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"Error finding document '{article_id}' in Firestore: {e}")
            return None

    def save_summary(self, summary: Summary):
        if not summaries_collection: return
        try:
            summaries_collection.document(summary.article_id).set(asdict(summary))
            logger.info(f"Saved summary for article ID {summary.article_id} to Firestore.")
        except Exception as e:
            logger.error(f"Error saving document for article ID {summary.article_id} to Firestore: {e}")


# --- NewsAggregator (Fetches news, currently using mock data) ---
class NewsAggregator:
    def fetch_financial_news(self, limit: int = 3) -> List[NewsArticle]:
        mock_articles_data = [
            {"id": "fed-rates-decision-2024", "title": "Federal Reserve Announces Interest Rate Decision", "content": "The Federal Reserve announced today that it will maintain current interest rates at 5.25-5.5% range, citing ongoing concerns about inflation and employment data. The decision comes after weeks of speculation about potential rate cuts. Fed Chair Jerome Powell emphasized the committee's commitment to bringing inflation down to the 2% target while maintaining a strong labor market. Economic indicators show mixed signals, with unemployment at 3.7% and core inflation at 3.2%. The decision affects mortgage rates, business lending, and overall economic growth.", "url": "https://example.com/fed-rates", "published_at": datetime.now() - timedelta(hours=2), "source": "Financial Times", "category": "monetary_policy"},
            {"id": "tesla-q4-earnings-2024", "title": "Tesla Stock Surges 15% on Q4 Earnings Beat", "content": "Tesla Inc. shares jumped 15% in after-hours trading following the company's stronger-than-expected Q4 earnings report. The electric vehicle maker reported earnings per share of $0.71, beating analyst estimates of $0.63. Revenue reached $25.2 billion, up 3% year-over-year. CEO Elon Musk highlighted improved production efficiency and strong demand in China and Europe. The company delivered 484,507 vehicles in Q4.", "url": "https://example.com/tesla-earnings", "published_at": datetime.now() - timedelta(hours=1), "source": "Reuters", "category": "earnings"},
            {"id": "bitcoin-ath-2024", "title": "Bitcoin Hits New All-Time High Above $73,000", "content": "Bitcoin reached a new all-time high of $73,147 today, driven by increased institutional adoption and speculation about potential Bitcoin ETF approvals. The cryptocurrency has gained over 160% year-to-date. Major factors include MicroStrategy's additional $1.5 billion Bitcoin purchase. However, analysts warn of potential volatility.", "url": "https://example.com/bitcoin-ath", "published_at": datetime.now() - timedelta(minutes=30), "source": "CoinDesk", "category": "cryptocurrency"}
        ]
        return [NewsArticle(**data) for data in mock_articles_data[:limit]]


# --- AIProcessor (Handles interaction with the Groq LLM) ---
class AIProcessor:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        if not self.client:
            logger.warning("GROQ_API_KEY not found. AI features will be disabled.")
        self.model = "llama3-8b-8192"

    def generate_summary(self, article: NewsArticle, user_level: str) -> Summary:
        if not self.client:
            return Summary(article_id=article.id, simple_summary="AI processor is offline.", key_points=[], impact_analysis="", investment_implications="", difficulty_level="N/A", reading_time_minutes=1)
        
        prompt = f"""
        You are an expert financial analyst who simplifies complex news for investors.
        Summarize the following financial news article for an investor with a '{user_level}' experience level.

        Article Title: "{article.title}"
        Article Content: "{article.content}"

        Your task is to return a JSON object with the following exact structure:
        {{
          "article_id": "{article.id}",
          "simple_summary": "A very simple, one-paragraph explanation of the news. Write it as if explaining to a friend.",
          "key_points": ["A list of 3-4 most important bullet points from the article."],
          "impact_analysis": "A brief analysis of what this news could mean for the market or the specific company/asset. Explain the 'so what?'.",
          "investment_implications": "Provide a short, balanced view on potential investment considerations. Do not give direct financial advice. Use phrases like 'Investors might consider...' or 'This could be positive/negative for...'.",
          "difficulty_level": "Classify the topic's complexity as 'beginner', 'intermediate', or 'advanced'.",
          "reading_time_minutes": 2
        }}

        Do not include any introductory text. Only return the valid JSON.
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            summary_data = json.loads(chat_completion.choices[0].message.content)
            return Summary(**summary_data)
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return Summary(article_id=article.id, simple_summary="AI summary could not be generated at this time.", key_points=[], impact_analysis="N/A", investment_implications="N/A", difficulty_level="N/A", reading_time_minutes=1)

    def generate_daily_brief(self, articles: List[NewsArticle], user_assets: List[str]) -> Dict:
        if not self.client:
            return {"date": datetime.now().strftime("%B %d, %Y"), "market_overview": "AI processor is offline.", "key_themes": [], "tomorrow_watch": []}

        article_titles = "\n- ".join([f'"{a.title}"' for a in articles])
        assets_str = ", ".join(user_assets) if user_assets else "general market"
        
        prompt = f"""
        You are a financial news editor for an app called FinanceFlow. Your task is to write a concise daily market briefing for a user interested in: {assets_str}.

        Today's key news headlines are:
        - {article_titles}

        Based on these headlines, generate a JSON object with the following structure:
        {{
          "date": "{datetime.now().strftime("%B %d, %Y")}",
          "market_overview": "A one-paragraph summary of the overall market sentiment today based on the headlines. Is it positive, negative, or mixed?",
          "key_themes": ["A list of 2-3 major themes or trends observed from today's news."],
          "tomorrow_watch": ["A list of 2-3 things investors should watch out for tomorrow, based on today's events."]
        }}
        
        Do not include any introductory text. Only return the valid JSON.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            return json.loads(chat_completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"Groq API call for daily brief failed: {e}")
            return {"date": datetime.now().strftime("%B %d, %Y"), "market_overview": "Could not generate brief.", "key_themes": [], "tomorrow_watch": []}


# --- Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# Initialize components
firestore_manager = FirestoreManager()
news_fetcher = NewsAggregator()
ai_processor = AIProcessor()

# Read the frontend file into a variable once at startup
try:
    with open('finance_news_frontend.html', 'r', encoding='utf-8') as f:
        FRONTEND_HTML = f.read()
except FileNotFoundError:
    logger.error("FATAL: finance_news_frontend.html not found. App will not serve frontend.")
    FRONTEND_HTML = "<h1>Error: Frontend file not found. Please check deployment.</h1>"


# --- API Routes ---
@app.route('/')
def index():
    return render_template_string(FRONTEND_HTML)

@app.route('/api/news')
def get_news():
    try:
        level = request.args.get('level', 'beginner')
        articles = news_fetcher.fetch_financial_news()
        response_data = []
        
        for article in articles:
            summary = firestore_manager.find_summary_by_article_id(article.id)
            if not summary:
                logger.info(f"Cache miss for article ID {article.id}. Calling Groq API.")
                summary = ai_processor.generate_summary(article, level)
                firestore_manager.save_summary(summary)
            
            article_data = asdict(article)
            article_data['published_at'] = article.published_at.isoformat()
            response_data.append({'article': article_data, 'summary': asdict(summary)})
            
        return jsonify({'status': 'success', 'data': response_data})
    except Exception as e:
        logger.error(f"Error in /api/news: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/api/daily-brief')
def get_daily_brief():
    try:
        assets_str = request.args.get('assets', 'tesla,bitcoin')
        user_assets = assets_str.split(',') if assets_str else []
        articles = news_fetcher.fetch_financial_news()
        brief = ai_processor.generate_daily_brief(articles, user_assets)
        return jsonify({'status': 'success', 'data': brief})
    except Exception as e:
        logger.error(f"Error in /api/daily-brief: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# This block is for local development only.
# Gunicorn, the production server on Render, will not run this.
if __name__ == '__main__':
    print("🚀 Starting FinanceFlow [LOCAL DEVELOPMENT MODE WITH FIREBASE]")
    app.run(debug=True, host='0.0.0.0', port=5000)