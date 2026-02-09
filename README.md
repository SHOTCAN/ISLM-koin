# ISLM Monitor V3 - Haqq Network Specialist 🌙🏰
![Visitor Count](https://profile-counter.glitch.me/SHOTCAN-ISLM-koin/count.svg)

Real-time ISLM/IDR price monitor with AI-powered analytics, interactive Telegram bot, and military-grade security.

## Features
- 📊 Real-time candlestick chart (1m / 15m / 1H / 1D)
- 🧠 AI Signal Engine with explainable reasoning
- 🔮 Multi-horizon price prediction (1 day / 3 day / 7 day)
- 🐋 Whale tracker & order book analysis
- 🕯️ Candlestick pattern detection
- 💬 Context-aware AI chatbot
- 📱 Interactive Telegram bot with buttons
- 🔐 2FA authentication via Telegram OTP

## Setup

1. Clone the repository
2. Configure secrets in `.streamlit/secrets.toml` or `.env`:
```
INDODAX_API_KEY=your_key
INDODAX_SECRET_KEY=your_secret
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```
3. Install dependencies:
```
pip install -r requirements.txt
```
4. Run:
```
streamlit run app_web.py
```

## Security
- All API keys stored in environment variables
- Telegram OTP 2-Factor Authentication
- Session timeout (15 minutes)
- Login attempt limiting (3 max)
- No credentials in repository

## Tech Stack
- Python 3.10+
- Streamlit (Web UI)
- Plotly (Interactive Charts)
- NumPy/Pandas (Analytics)
- Telegram Bot API (Notifications)
- Indodax API (Market Data)
