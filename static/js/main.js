// static/js/main.js

document.addEventListener('DOMContentLoaded', initializeApp);

// --- Global State ---
let newsData = [];
let stockData = {};
let isAsking = false; // To prevent multiple simultaneous questions

// --- DOM Elements ---
const stockBarContainer = document.getElementById('stock-bar-container');
const newsContainer = document.getElementById('news-container');
const qaInput = document.getElementById('qa-input');
const qaButton = document.getElementById('qa-button');
const qaAnswerBox = document.getElementById('qa-answer-box');

// --- Initialization ---
function initializeApp() {
    fetchMainFeed();
    qaInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            handleAskQuestion();
        }
    });
}

// --- API Calls ---
async function fetchMainFeed() {
    try {
        const response = await fetch('/api/main_feed');
        if (!response.ok) {
            throw new Error(`Network response was not ok: ${response.statusText}`);
        }
        const result = await response.json();
        if (result.status === 'success') {
            newsData = result.data.news;
            stockData = result.data.stocks;
            renderAll();
        } else {
            renderError(result.message || 'Failed to load data.');
        }
    } catch (error) {
        console.error('Failed to fetch main feed:', error);
        renderError('Could not connect to the server. Please try again later.');
    }
}

async function handleAskQuestion() {
    const question = qaInput.value.trim();
    if (!question || isAsking) return;

    isAsking = true;
    qaButton.disabled = true;
    qaButton.textContent = 'Thinking...';
    qaAnswerBox.style.opacity = 0.5;
    qaAnswerBox.textContent = 'AI is processing your question...';

    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });
        const result = await response.json();
        if (result.status === 'success') {
            qaAnswerBox.textContent = result.answer;
        } else {
            qaAnswerBox.textContent = `Error: ${result.message}`;
        }
    } catch (error) {
        console.error('Failed to ask question:', error);
        qaAnswerBox.textContent = 'Sorry, an error occurred while asking the AI.';
    } finally {
        isAsking = false;
        qaButton.disabled = false;
        qaButton.textContent = 'Ask AI';
        qaAnswerBox.style.opacity = 1;
        qaInput.value = '';
    }
}

// --- Rendering Functions ---
function renderAll() {
    renderStockBar();
    renderNewsFeed();
}

function renderStockBar() {
    const stockSymbols = Object.keys(stockData);
    if (stockSymbols.length === 0) {
        stockBarContainer.innerHTML = '<div class="stock-box">Fetching stock data...</div>';
        return;
    }

    const stockHTML = stockSymbols.map(symbol => {
        const stock = stockData[symbol];
        const changeClass = stock.change >= 0 ? 'positive' : 'negative';
        const sign = stock.change >= 0 ? '+' : '';
        return `
            <div class="stock-box">
                <span class="symbol">${stock.symbol}</span>
                <span class="price">${stock.price.toFixed(2)}</span>
                <span class="change ${changeClass}">
                    ${sign}${stock.change.toFixed(2)} (${sign}${stock.percent_change.toFixed(2)}%)
                </span>
            </div>
        `;
    }).join('');
    
    // Duplicate the content for a seamless looping animation
    stockBarContainer.innerHTML = `<div class="stock-ticker-inner">${stockHTML}${stockHTML}</div>`;
}

function renderNewsFeed() {
    if (newsData.length === 0) {
        renderError('No news available at the moment. The worker might be processing data in the background.');
        return;
    }
    newsContainer.innerHTML = newsData.map(renderNewsCard).join('');
}

function renderNewsCard(item) {
    // Ensure analysis object exists to prevent errors
    const analysis = item.analysis || {};
    const sentiment = analysis.sentiment || 'Neutral';
    const impact_score = analysis.impact_score || 'N/A';
    const summary = analysis.summary_th || analysis.summary_en || 'Summary not available.';
    
    const sentimentClass = `sentiment-${sentiment}`;
    const publishedDate = new Date(item.published).toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' });

    return `
        <div class="story-card">
            <header class="card-header">
                <span class="card-source">${item.source}</span>
                <time datetime="${item.published}">${publishedDate}</time>
            </header>
            <h2 class="card-title">${item.title}</h2>
            <p class="card-summary">${summary}</p>
            <footer class="card-footer">
                <div class="sentiment-badge ${sentimentClass}">${sentiment}</div>
                <div class="impact-score">Impact: ${impact_score}/10</div>
                <a href="${item.link}" target="_blank" rel="noopener noreferrer" class="source-link">Read Source →</a>
            </footer>
        </div>
    `;
}

function renderError(message) {
    newsContainer.innerHTML = `
        <div class="error-screen">
            <h2>Something went wrong</h2>
            <p>${message}</p>
        </div>
    `;
}