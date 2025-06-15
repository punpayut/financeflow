"""
FinanceFlow - Financial News Summarizer (Powered by Groq with .env)
A comprehensive financial news summarization platform for beginner-intermediate investors
"""
import json
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests
from dataclasses import dataclass, asdict
from groq import Groq
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging
from dotenv import load_dotenv

# Load variables from .env file. This should be at the top.
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Data Classes ---
@dataclass
class NewsArticle:
    """Data structure for news articles"""
    id: str
    title: str
    content: str
    url: str
    published_at: datetime
    source: str
    category: str = "general"
    sentiment: str = "neutral"
    
@dataclass
class Summary:
    """Data structure for article summaries"""
    article_id: str
    simple_summary: str
    key_points: List[str]
    impact_analysis: str
    investment_implications: str
    difficulty_level: str
    reading_time_minutes: int

# --- DatabaseManager ---
class DatabaseManager:
    """Handles all database operations"""
    def __init__(self, db_path: str = "financeflow.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS articles (id TEXT PRIMARY KEY, title TEXT, content TEXT, url TEXT, published_at TIMESTAMP, source TEXT, category TEXT, sentiment TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS summaries (id INTEGER PRIMARY KEY, article_id TEXT, simple_summary TEXT, key_points TEXT, impact_analysis TEXT, investment_implications TEXT, difficulty_level TEXT, reading_time_minutes INTEGER, FOREIGN KEY (article_id) REFERENCES articles (id))''')
        conn.commit()
        conn.close()

# --- NewsAggregator (Using mock data) ---
class NewsAggregator:
    """Fetches news from various sources"""
    def fetch_financial_news(self, category: str = "business", limit: int = 20) -> List[NewsArticle]:
        mock_articles_data = [
            {"id": "1", "title": "Federal Reserve Announces Interest Rate Decision", "content": "The Federal Reserve announced today that it will maintain current interest rates at 5.25-5.5% range, citing ongoing concerns about inflation and employment data. The decision comes after weeks of speculation about potential rate cuts. Fed Chair Jerome Powell emphasized the committee's commitment to bringing inflation down to the 2% target while maintaining a strong labor market. Economic indicators show mixed signals, with unemployment at 3.7% and core inflation at 3.2%. The decision affects mortgage rates, business lending, and overall economic growth.", "url": "https://example.com/fed-rates", "published_at": datetime.now() - timedelta(hours=2), "source": "Financial Times", "category": "monetary_policy"},
            {"id": "2", "title": "Tesla Stock Surges 15% on Q4 Earnings Beat", "content": "Tesla Inc. shares jumped 15% in after-hours trading following the company's stronger-than-expected Q4 earnings report. The electric vehicle maker reported earnings per share of $0.71, beating analyst estimates of $0.63. Revenue reached $25.2 billion, up 3% year-over-year. CEO Elon Musk highlighted improved production efficiency and strong demand in China and Europe. The company delivered 484,507 vehicles in Q4.", "url": "https://example.com/tesla-earnings", "published_at": datetime.now() - timedelta(hours=1), "source": "Reuters", "category": "earnings"},
            {"id": "3", "title": "Bitcoin Hits New All-Time High Above $73,000", "content": "Bitcoin reached a new all-time high of $73,147 today, driven by increased institutional adoption and speculation about potential Bitcoin ETF approvals. The cryptocurrency has gained over 160% year-to-date. Major factors include MicroStrategy's additional $1.5 billion Bitcoin purchase. However, analysts warn of potential volatility.", "url": "https://example.com/bitcoin-ath", "published_at": datetime.now() - timedelta(minutes=30), "source": "CoinDesk", "category": "cryptocurrency"}
        ]
        return [NewsArticle(**data) for data in mock_articles_data[:limit]]

# --- AIProcessor (Using Groq API) ---
class AIProcessor:
    """Handles AI-powered summarization and analysis using Groq API"""
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in .env file or environment. AI features will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
        self.model = "llama3-8b-8192"

    def _create_chat_completion(self, prompt: str) -> Optional[dict]:
        """Helper function to call Groq API and parse JSON response."""
        if not self.client: return None
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.2,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            response_content = chat_completion.choices[0].message.content
            return json.loads(response_content)
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return None

    def generate_summary(self, article: NewsArticle, user_level: str = "beginner") -> Summary:
        prompt = f"""
        You are an expert financial analyst who simplifies complex news for investors.
        Summarize the following financial news article for an investor with a '{user_level}' experience level.

        Article Title: "{article.title}"
        Article Content: "{article.content}"

        Your task is to return a JSON object with the following exact structure:
        {{
          "simple_summary": "A very simple, one-paragraph explanation of the news. Write it as if explaining to a friend.",
          "key_points": ["A list of 3-4 most important bullet points from the article."],
          "impact_analysis": "A brief analysis of what this news could mean for the market or the specific company/asset. Explain the 'so what?'.",
          "investment_implications": "Provide a short, balanced view on potential investment considerations. Do not give direct financial advice. Use phrases like 'Investors might consider...' or 'This could be positive/negative for...'.",
          "difficulty_level": "Classify the topic's complexity as 'beginner', 'intermediate', or 'advanced'.",
          "reading_time_minutes": "Estimate the reading time of your summary in whole minutes (e.g., 2)."
        }}

        Do not include any introductory text. Only return the valid JSON.
        """
        summary_data = self._create_chat_completion(prompt)
        
        if not summary_data:
            return Summary(article_id=article.id, simple_summary="AI summary could not be generated at this time.", key_points=[], impact_analysis="", investment_implications="", difficulty_level="N/A", reading_time_minutes=1)
        
        return Summary(
            article_id=article.id,
            simple_summary=summary_data.get("simple_summary", "Summary not available."),
            key_points=summary_data.get("key_points", []),
            impact_analysis=summary_data.get("impact_analysis", "Impact analysis not available."),
            investment_implications=summary_data.get("investment_implications", "Implications not available."),
            difficulty_level=summary_data.get("difficulty_level", "intermediate"),
            reading_time_minutes=summary_data.get("reading_time_minutes", 2)
        )

    def generate_daily_brief(self, articles: List[NewsArticle], user_assets: List[str]) -> Dict:
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
        brief_data = self._create_chat_completion(prompt)
        
        if not brief_data:
            return {"date": datetime.now().strftime("%B %d, %Y"), "market_overview": "Could not generate market overview.", "key_themes": [], "tomorrow_watch": []}
        
        return brief_data

# --- Flask Application Setup ---
app = Flask(__name__)
CORS(app)

# Initialize components
db = DatabaseManager()
news_fetcher = NewsAggregator()
ai_processor = AIProcessor()

try:
    with open('finance_news_frontend.html', 'r', encoding='utf-8') as f:
        FRONTEND_HTML = f.read()
except FileNotFoundError:
    logger.error("FATAL: finance_news_frontend.html not found. Make sure it's in the same directory.")
    FRONTEND_HTML = "<h1>Error: Frontend file not found.</h1>"

# --- API Routes ---
@app.route('/')
def index():
    return render_template_string(FRONTEND_HTML)

@app.route('/api/news')
def get_news():
    try:
        level = request.args.get('level', 'beginner')
        articles = news_fetcher.fetch_financial_news(limit=3)
        response_data = []
        for article in articles:
            summary = ai_processor.generate_summary(article, level)
            article_data = asdict(article)
            article_data['published_at'] = article.published_at.isoformat()
            response_data.append({'article': article_data, 'summary': asdict(summary)})
        return jsonify({'status': 'success', 'data': response_data})
    except Exception as e:
        logger.error(f"Error in /api/news: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/api/daily-brief')
def get_daily_brief():
    try:
        assets_str = request.args.get('assets', 'tesla,bitcoin')
        user_assets = assets_str.split(',') if assets_str else []
        articles = news_fetcher.fetch_financial_news(limit=5)
        brief = ai_processor.generate_daily_brief(articles, user_assets)
        return jsonify({'status': 'success', 'data': brief})
    except Exception as e:
        logger.error(f"Error in /api/daily-brief: {str(e)}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting FinanceFlow - Financial News Summarizer (Groq Edition with .env)")
    print("📊 AI-powered financial news made simple")
    print("🌐 Access the application at: http://127.0.0.1:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)