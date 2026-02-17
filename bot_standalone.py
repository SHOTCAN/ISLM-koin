"""
ISLM Monitor — Standalone Telegram AI Bot V7
=============================================
Jalan INDEPENDEN dari Streamlit. Deploy ke Railway = 24/7.

V7 Features:
  - AI Genius Mode + Memory + Chain-of-Thought
  - Smart Price Alerts (±3% auto-notify)
  - Volatility Adaptive System (10min during turbulence)
  - Image Analysis (Groq Vision)
  - Phone Number Lookup
  - Groq Rate Limiter + Anti-spam
"""

import sys
import os

# Fix Windows console encoding BEFORE any emoji prints
os.environ['PYTHONIOENCODING'] = 'utf-8'

import time
import traceback
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta


def safe_print(*args, **kwargs):
    """Print that won't crash on Windows with emoji."""
    try:
        print(*args, **kwargs, flush=True)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode('ascii', 'replace').decode(), flush=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backend.api import IndodaxAPI
from backend.config import Config
from backend.core_logic import (
    MarketProjector, FundamentalEngine, QuantAnalyzer,
    CandleSniper, WhaleTracker, AISignalEngine,
    ProTA, MLSignalClassifier, SupportResistance, NewsEngine,
    BTCCorrelation, RiskMetrics, TrendStrengthIndex,
)
from backend.security_engine import SecurityEngine

import requests


# ============================================
# ============================================
# HEALTH CHECK SERVER (Railway IPv6 compatible)
# ============================================
import socket

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'ISLM Bot V7 OK')
    def log_message(self, *a): pass

class IPv6HTTPServer(HTTPServer):
    address_family = socket.AF_INET6

def start_health_server():
    port = int(os.environ.get('PORT', 8000))
    try:
        server = IPv6HTTPServer(('::', port), HealthHandler)
        safe_print(f"[HEALTH] HTTP server on [::] port {port} (IPv6)")
    except Exception:
        server = HTTPServer(('0.0.0.0', port), HealthHandler)
        safe_print(f"[HEALTH] HTTP server on 0.0.0.0 port {port} (IPv4 fallback)")
    threading.Thread(target=server.serve_forever, daemon=True).start()


# ============================================
# CONFIG
# ============================================
INTERVAL_30MIN = 1800   # 30 minutes
INTERVAL_1HOUR = 3600   # 1 hour
INTERVAL_DAILY = 86400  # 24 hours
POLL_INTERVAL = 3       # Telegram poll every 3s
PAIR = 'islmidr'


# ============================================
# STANDALONE BOT V4
# ============================================
class StandaloneBot:
    def __init__(self):
        # Try Config first, then direct os.environ (Railway injects env vars)
        self.token = Config.TELEGRAM_TOKEN or os.environ.get('TELEGRAM_TOKEN', '')
        self.chat_id = Config.TELEGRAM_CHAT_ID or os.environ.get('TELEGRAM_CHAT_ID', '')

        if self.token:
            safe_print(f"[CONFIG] Token loaded OK")
        else:
            safe_print("[CONFIG] WARNING: No TELEGRAM_TOKEN found!")
            safe_print(f"[CONFIG] Telegram env keys: {[k for k in os.environ if 'TELEGRAM' in k.upper()]}")

        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = None
        self.api = IndodaxAPI(
            Config.API_KEY or os.environ.get('INDODAX_API_KEY', ''),
            Config.SECRET_KEY or os.environ.get('INDODAX_SECRET_KEY', '')
        )
        self.security = SecurityEngine(self.token, self.chat_id)

        # Interval timers
        self.last_30min = 0
        self.last_1hour = 0
        self.last_daily = 0
        self.start_time = time.time()

        # Price history for daily recap
        self.price_history = []
        self.daily_high = 0
        self.daily_low = float('inf')

        # V6: Smart Alert System
        self.last_alert_time = 0
        self.alert_cooldown = 900      # 15 min minimum between alerts
        self.alert_threshold = 0.03    # 3% price change triggers alert
        self.alerts_this_hour = 0
        self.alerts_hour_reset = time.time()
        self.max_alerts_per_hour = 4
        self.last_alert_price = 0

        # V6: Volatility Adaptive System
        self.volatility_mode = False
        self.volatile_interval = 600   # 10 min during high volatility
        self.last_volatile_check = 0
        self.price_samples_5min = []   # Recent samples for volatility calc

        # V6: Conversation memory
        self._chat_memory = []

        # V7: Groq Rate Limiter
        self._groq_requests = []  # timestamps of recent requests
        self._groq_rpm_limit = 28  # stay under 30 RPM free tier
        self._groq_daily_count = 0
        self._groq_daily_reset = time.time()

        # AI cache
        self._cache = {
            "price": 0, "rsi": 50, "macd_val": 0, "hist": 0, "sig_line": 0,
            "bb_upper": None, "bb_mid": None, "bb_lower": None,
            "whale_ratio": 0.5, "whale_label": "⚖️ Whale Seimbang",
            "ai_signal": {"label": "HOLD 🤝", "confidence": 0, "trend": "Sideways", "reasons": [], "score": 0},
            "predictions": {}, "candle_patterns": [],
            "f_score": 0, "f_news": "", "market_phase": "NETRAL",
            "atr": None, "stoch_k": 50,
            "pro_ta": {}, "ml_result": {},
            "supports": [], "resistances": [],
            "last_update": "Never",
        }

        # V7: Portfolio Tracker
        self.portfolio = []  # [{"buy_price": X, "amount": Y, "date": Z}]
        self.total_invested = 0

        # V7: Custom Price Targets
        self.price_targets = []  # [{"price": X, "direction": "above"/"below", "notified": False}]

        # V7: Bot Stats
        self._stats = {
            "messages_handled": 0,
            "ai_calls": 0,
            "alerts_sent": 0,
            "photos_analyzed": 0,
            "errors": 0,
        }

        # V7: Notification Control
        self.notifications_paused = False

        # V7: GitHub Repo Security
        self._github_repo = os.environ.get('GITHUB_REPO', 'SHOTCAN/ISLM-koin')
        self._github_token = os.environ.get('GITHUB_TOKEN', '')
        self._last_repo_check = 0
        self._repo_check_interval = 1800  # 30 min
        self._last_known_watchers = -1
        self._last_known_forks = -1
        self._last_known_stars = -1

    # --- Telegram ---
    def send_message(self, text, parse_mode="Markdown"):
        if not self.chat_id: return False
        try:
            for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
                r = requests.post(f"{self.base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": chunk, "parse_mode": parse_mode}, timeout=10)
            return True
        except Exception as e:
            safe_print(f"[SEND ERR] {e}")
            return False

    def send_message_with_keyboard(self, text, buttons, parse_mode="Markdown"):
        """Send message with inline keyboard buttons."""
        if not self.chat_id: return False
        try:
            keyboard = {"inline_keyboard": buttons}
            r = requests.post(f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "reply_markup": keyboard
                }, timeout=10)
            return True
        except Exception as e:
            safe_print(f"[SEND KB ERR] {e}")
            return False

    def get_updates(self):
        try:
            r = requests.get(f"{self.base_url}/getUpdates",
                params={'timeout': POLL_INTERVAL, 'offset': self.offset}, timeout=POLL_INTERVAL + 5)
            return r.json()
        except:
            return {}

    # ============================================
    # CORE: Full Analysis Pipeline
    # ============================================
    def refresh_analysis(self):
        try:
            # Price
            ticker = self.api.get_price(PAIR)
            if not ticker.get('success'):
                safe_print("[WARN] Price fetch failed")
                return False
            price = ticker['last']

            # Track daily high/low
            self.price_history.append(price)
            self.daily_high = max(self.daily_high, price)
            self.daily_low = min(self.daily_low, price)

            # Candles
            candles = self.api.get_kline(PAIR, '15')
            if not candles or len(candles) < 10:
                return False

            df = pd.DataFrame(candles)
            closes = df['close'].values
            highs_arr = df['high'].values
            lows_arr = df['low'].values

            # --- Manual TA (fallback) ---
            rsi = QuantAnalyzer.calculate_rsi(closes)
            macd_val, sig_line, hist = QuantAnalyzer.calculate_macd(closes)
            stoch_k, _ = QuantAnalyzer.calculate_stoch_rsi(closes)
            bb_upper, bb_mid, bb_lower = QuantAnalyzer.calculate_bollinger_bands(closes)
            atr = QuantAnalyzer.calculate_atr(highs_arr, lows_arr, closes)
            market_phase = QuantAnalyzer.detect_market_phase(closes)

            # --- ProTA (40+ indicators from ta library) ---
            pro_ta = ProTA.compute_all(df)

            # --- ML Signal ---
            ml_result = MLSignalClassifier.predict_signal(df)

            # --- Support/Resistance ---
            supports, resistances = SupportResistance.find_levels(df)

            # Override with ProTA values if available
            if pro_ta.get('rsi'): rsi = pro_ta['rsi']
            if pro_ta.get('macd'): macd_val = pro_ta['macd']
            if pro_ta.get('macd_signal'): sig_line = pro_ta['macd_signal']
            if pro_ta.get('macd_hist'): hist = pro_ta['macd_hist']
            if pro_ta.get('bb_upper'): bb_upper = pro_ta['bb_upper']
            if pro_ta.get('bb_mid'): bb_mid = pro_ta['bb_mid']
            if pro_ta.get('bb_lower'): bb_lower = pro_ta['bb_lower']
            if pro_ta.get('atr'): atr = pro_ta['atr']
            if pro_ta.get('stoch_k'): stoch_k = pro_ta['stoch_k']

            # Whale
            try:
                depth = self.api.get_depth(PAIR) or {}
                whale_ratio = WhaleTracker.get_whale_ratio(depth.get('buy', []), depth.get('sell', []), 0.1)
            except:
                whale_ratio = 0.5
            whale_label = WhaleTracker.interpret(whale_ratio)

            # Candle Patterns
            candle_patterns = CandleSniper.analyze_patterns(candles)
            bull_k = ("HAMMER", "INV. HAMMER", "BULL ENGULFING", "MORNING STAR")
            bear_k = ("HANGING MAN", "SHOOTING STAR", "BEAR ENGULFING", "EVENING STAR")
            cb = sum(1 for p in candle_patterns if any(k in p for k in bull_k))
            cbe = sum(1 for p in candle_patterns if any(k in p for k in bear_k))

            # Fundamental
            f_score, f_news = FundamentalEngine.analyze_market_sentiment()

            # AI Signal V4
            ai_signal = AISignalEngine.compute(
                rsi=rsi, macd_hist=hist, price=price,
                bb_mid=bb_mid, bb_upper=bb_upper, bb_lower=bb_lower,
                candle_bull_count=cb, candle_bear_count=cbe,
                whale_ratio=whale_ratio, fundamental_score=f_score,
                pro_ta=pro_ta, ml_result=ml_result,
            )

            # Predictions
            price_list = [c['close'] for c in candles[-100:]]
            predictions = MarketProjector.predict_multi_horizon(price, price_list)

            # Update cache
            self._cache.update({
                "price": price, "rsi": rsi, "macd_val": macd_val, "hist": hist, "sig_line": sig_line,
                "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
                "whale_ratio": whale_ratio, "whale_label": whale_label,
                "ai_signal": ai_signal, "predictions": predictions,
                "candle_patterns": candle_patterns,
                "f_score": f_score, "f_news": f_news, "market_phase": market_phase,
                "atr": atr, "stoch_k": stoch_k,
                "pro_ta": pro_ta, "ml_result": ml_result,
                "supports": supports, "resistances": resistances,
                "last_update": datetime.now().strftime('%H:%M:%S'),
            })

            safe_print(f"[OK] V4 Analysis | Rp {price:,.0f} | {ai_signal['label']} | ML: {ml_result.get('ml_signal', 'N/A')}")
            return True

        except Exception as e:
            safe_print(f"[ERROR] Analysis: {e}")
            traceback.print_exc()
            return False

    # ============================================
    # NOTIFICATION: 30 Menit — Quick Status
    # ============================================
    def send_30min_update(self):
        c = self._cache
        sig = c["ai_signal"]
        news = NewsEngine.generate_news(c['market_phase'], c['whale_ratio'])

        # Generate AI mini-analysis via Groq
        ai_insight = ""
        try:
            api_key = Config.GROQ_API_KEY if hasattr(Config, 'GROQ_API_KEY') else ''
            if api_key:
                from groq import Groq
                client = Groq(api_key=api_key)

                brief_context = (
                    f"ISLM Rp {c['price']:,.0f}, RSI={c['rsi']:.0f}, "
                    f"MACD hist={c['hist']:+.2f}, Signal={sig['label']}, "
                    f"Trend={sig['trend']}, Phase={c['market_phase']}, "
                    f"Whale={c['whale_label']}"
                )
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content":
                            "Kamu AI trading analyst. Berikan analisa SINGKAT (max 2 kalimat) "
                            "tentang kondisi ISLM saat ini dan saran aksi (hold/buy/sell/wait). "
                            "Bahasa Indonesia santai. Sertakan alasan singkat."},
                        {"role": "user", "content": brief_context}
                    ],
                    max_tokens=150,
                    temperature=0.7,
                )
                ai_insight = resp.choices[0].message.content.strip()
        except:
            ai_insight = ""

        msg = (
            f"⚡ *ISLM UPDATE (30min)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Harga: Rp {c['price']:,.0f}\n"
            f"📢 Sinyal: {sig['label']} ({sig['confidence']*100:.0f}%)\n"
            f"📈 Trend: {sig['trend']}\n"
            f"📊 RSI: {c['rsi']:.1f} | MACD: {c['hist']:+.2f}\n"
            f"🐋 Whale: {c['whale_label']}\n"
        )

        if ai_insight:
            msg += f"\n🧠 *AI Insight:*\n{ai_insight}\n"

        msg += f"\n📰 *News:* {news}\n"
        msg += f"⏰ {c['last_update']} WIB"

        self.send_message(msg)

    # ============================================
    # NOTIFICATION: 1 Jam — Full Analysis V5
    # ============================================
    def send_1hour_update(self):
        c = self._cache
        sig = c["ai_signal"]
        ml = c["ml_result"]
        pta = c["pro_ta"]

        lines = [
            "🤖 *ISLM FULL ANALYSIS V5 (1 Jam)*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 *Harga:* Rp {c['price']:,.0f}",
            f"📢 *AI Signal:* {sig['label']} ({sig['confidence']*100:.0f}%)",
        ]

        if ml.get('ml_available'):
            lines.append(f"🤖 *ML Model:* {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)")

        lines += [
            f"📈 *Trend:* {sig['trend']}",
            f"🏷️ *Fase:* {c['market_phase']}",
            "",
            "📊 *INDIKATOR:*",
            f"  RSI: {c['rsi']:.1f} | MACD: {c['macd_val']:.2f}",
            f"  Stoch: {c['stoch_k']:.1f}",
        ]

        if c['atr']:
            lines.append(f"  ATR: {c['atr']:.2f}")
        if c['bb_upper']:
            lines.append(f"  BB: Rp {c['bb_lower']:,.0f} — Rp {c['bb_upper']:,.0f}")
        if pta.get('adx'):
            lines.append(f"  ADX: {pta['adx']:.0f}")
        if pta.get('mfi'):
            lines.append(f"  MFI: {pta['mfi']:.0f}")

        # V5: Risk Metrics
        if len(self.price_history) >= 10:
            try:
                risk = RiskMetrics.full_report(self.price_history)
                lines.append(f"\n📊 *RISK:*")
                lines.append(f"  Sharpe: {risk['sharpe']:.2f} | MaxDD: {risk['max_dd']:.1f}%")
                lines.append(f"  WinRate: {risk['win_rate']:.0f}% | VaR: {risk['var_95']:.2f}%")
            except: pass

        # Support/Resistance
        if c['supports']:
            lines.append(f"\n🟢 *Support:* {', '.join(f'Rp {s:,.0f}' for s in c['supports'])}")
        if c['resistances']:
            lines.append(f"🔴 *Resistance:* {', '.join(f'Rp {r:,.0f}' for r in c['resistances'])}")

        # Predictions
        lines.append("\n🔮 *PREDIKSI:*")
        for k, p in c["predictions"].items():
            lines.append(f"  {p['label']}: Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%) {p['direction']}")

        lines.append(f"\n🐋 {c['whale_label']}")
        if c['candle_patterns']:
            lines.append(f"🕯️ Pola: {', '.join(c['candle_patterns'])}")

        # Top 5 reasons
        lines.append("\n🧠 *REASONING:*")
        for r in sig.get("reasons", [])[:5]:
            lines.append(f"  • {r}")

        # V5: AI Strategy via Groq
        try:
            api_key = Config.GROQ_API_KEY if hasattr(Config, 'GROQ_API_KEY') else ''
            if api_key:
                from groq import Groq
                client = Groq(api_key=api_key)
                brief = (
                    f"ISLM Rp {c['price']:,.0f}, RSI={c['rsi']:.0f}, MACD={c['hist']:+.2f}, "
                    f"Signal={sig['label']}, Trend={sig['trend']}, Phase={c['market_phase']}, "
                    f"Whale={c['whale_label']}, Predictions: "
                    + ", ".join(f"{p['label']}={p['direction']}" for p in c['predictions'].values())
                )
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content":
                            "Kamu AI trading strategist. Berikan STRATEGI AKSI 1 jam ke depan. "
                            "Max 3 kalimat: (1) kondisi saat ini, (2) aksi yang disarankan, "
                            "(3) level kunci yang harus diperhatikan. Bahasa Indonesia."},
                        {"role": "user", "content": brief}
                    ],
                    max_tokens=200, temperature=0.7,
                )
                ai_strategy = resp.choices[0].message.content.strip()
                lines.append(f"\n🧠 *AI STRATEGY:*\n{ai_strategy}")
        except: pass

        lines.append(f"\n⏰ {c['last_update']} WIB | Next: 1h")
        self.send_message("\n".join(lines))

    # ============================================
    # NOTIFICATION: Harian — Daily Recap
    # ============================================
    def send_daily_update(self):
        c = self._cache
        sig = c["ai_signal"]
        ml = c["ml_result"]

        # Calculate daily stats
        if self.price_history:
            open_price = self.price_history[0]
            close_price = self.price_history[-1]
            daily_change = ((close_price - open_price) / (open_price + 1e-10)) * 100
            avg_price = sum(self.price_history) / len(self.price_history)
        else:
            open_price = close_price = avg_price = c['price']
            daily_change = 0

        emoji_change = "🟢" if daily_change >= 0 else "🔴"

        lines = [
            "📅 *ISLM DAILY RECAP V5*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"📆 {datetime.now().strftime('%d %B %Y')}",
            "",
            "💹 *RANGKUMAN HARI INI:*",
            f"  Open: Rp {open_price:,.0f}",
            f"  Close: Rp {close_price:,.0f}",
            f"  High: Rp {self.daily_high:,.0f}",
            f"  Low: Rp {self.daily_low:,.0f}",
            f"  Avg: Rp {avg_price:,.0f}",
            f"  {emoji_change} Change: {daily_change:+.2f}%",
            "",
            f"📢 *Sinyal AI:* {sig['label']} ({sig['confidence']*100:.0f}%)",
        ]

        if ml.get('ml_available'):
            lines.append(f"🤖 *ML Model:* {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)")

        lines += [
            f"📈 *Trend:* {sig['trend']}",
            f"🏷️ *Fase:* {c['market_phase']}",
        ]

        # V5: Risk Metrics
        if len(self.price_history) >= 10:
            try:
                risk = RiskMetrics.full_report(self.price_history)
                risk_level = '🟢 RENDAH' if risk['max_dd'] < 5 else '🟡 SEDANG' if risk['max_dd'] < 15 else '🔴 TINGGI'
                lines.append(f"\n📊 *RISK METRICS:*")
                lines.append(f"  Sharpe Ratio: {risk['sharpe']:.2f}")
                lines.append(f"  Max Drawdown: {risk['max_dd']:.1f}%")
                lines.append(f"  Win Rate: {risk['win_rate']:.0f}%")
                lines.append(f"  VaR (95%): {risk['var_95']:.2f}%")
                lines.append(f"  Level Risiko: {risk_level}")
            except: pass

        lines.append("\n🔮 *PREDIKSI BESOK:*")
        for k, p in c["predictions"].items():
            lines.append(
                f"  {p['label']}: Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%)\n"
                f"    Range: Rp {p['low']:,.0f} — Rp {p['high']:,.0f}"
            )

        if c['supports'] or c['resistances']:
            lines.append("\n📐 *LEVEL KUNCI:*")
            if c['supports']:
                lines.append(f"  🟢 Support: {', '.join(f'Rp {s:,.0f}' for s in c['supports'])}")
            if c['resistances']:
                lines.append(f"  🔴 Resistance: {', '.join(f'Rp {r:,.0f}' for r in c['resistances'])}")

        # V5: Security Status
        try:
            sec_status = self.security.get_security_status()
            lines.append(f"\n🛡️ *SECURITY:* {sec_status['threat_level']} | Threats: {sec_status['threat_count']}")
        except: pass

        # V5: AI Daily Summary via Groq
        try:
            api_key = Config.GROQ_API_KEY if hasattr(Config, 'GROQ_API_KEY') else ''
            if api_key:
                from groq import Groq
                client = Groq(api_key=api_key)
                daily_context = (
                    f"ISLM hari ini: Open Rp {open_price:,.0f}, Close Rp {close_price:,.0f}, "
                    f"High Rp {self.daily_high:,.0f}, Low Rp {self.daily_low:,.0f}, "
                    f"Change {daily_change:+.2f}%, Signal={sig['label']}, "
                    f"Trend={sig['trend']}, Phase={c['market_phase']}, "
                    f"Whale={c['whale_label']}, Samples={len(self.price_history)}"
                )
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content":
                            "Kamu AI trading analyst. Buat RINGKASAN HARIAN ISLM dalam 4 kalimat: "
                            "(1) Apa yang terjadi hari ini, (2) Faktor utama penggerak, "
                            "(3) Outlook untuk besok, (4) Saran strategi. "
                            "Bahasa Indonesia santai tapi profesional."},
                        {"role": "user", "content": daily_context}
                    ],
                    max_tokens=250, temperature=0.7,
                )
                ai_recap = resp.choices[0].message.content.strip()
                lines.append(f"\n🧠 *AI DAILY SUMMARY:*\n{ai_recap}")
        except: pass

        uptime_h = (time.time() - self.start_time) / 3600
        lines.append(f"\n⏱️ Uptime: {uptime_h:.1f} jam | Samples: {len(self.price_history)}")

        self.send_message("\n".join(lines))

        # Reset daily tracking
        self.price_history = []
        self.daily_high = 0
        self.daily_low = float('inf')

    # ============================================
    # MANUAL: Smart Intent Recognition
    # ============================================
    def handle_text_question(self, text):
        t = text.lower().strip()
        c = self._cache
        sig = c["ai_signal"]
        ml = c["ml_result"]
        pta = c["pro_ta"]

        # /start /menu /help
        if t in ['/start', '/menu', '/help', 'help', 'bantuan', 'menu']:
            return (
                "🤖 *ISLM AI V7 — MENU*\n\n"
                "*📊 Market:*\n"
                "  /status — Kondisi market\n"
                "  /predict — Prediksi harga\n"
                "  /analisa — Teknikal lengkap\n"
                "  /sinyal — Sinyal AI\n"
                "  /ml — Sinyal ML\n\n"
                "*🪙 Multi-Coin:*\n"
                "  /coin — Top coins Indodax\n"
                "  /coin btc — Harga BTC\n\n"
                "*📈 Advanced:*\n"
                "  /risk — Sharpe, MaxDD, WinRate\n"
                "  /level — Support & Resistance\n"
                "  /whale — Whale tracker\n"
                "  /candle — Pola candlestick\n"
                "  /news — Berita & fundamental\n\n"
                "*🆕 V7 Features:*\n"
                "  🖼️ Kirim gambar — AI analisa\n"
                "  /beli [harga] [jumlah] — Catat beli\n"
                "  /portfolio — Lihat P&L\n"
                "  /target [harga] — Alert di harga X\n"
                "  /stats — Statistik bot\n\n"
                "*🛡️ Security:*\n"
                "  /security — Status keamanan\n\n"
                "💬 *CHAT BEBAS:* Tanya apa saja!\n"
                "_Contoh: \"ISLM naik gak?\"_\n"
                "_Contoh: \"Bandingin BTC vs ISLM\"_"
            )
            # Also send inline keyboard
            # (this return text will be sent, then we optionally send buttons)

        # PHONE LOOKUP — V7
        if t.startswith('/cek') or t.startswith('cek '):
            parts = text.strip().split(maxsplit=1)
            if len(parts) >= 2:
                return self.lookup_phone(parts[1])
            return "📱 *Cara pakai:*\n/cek 08123456789\n/cek +628123456789"

        # V7: PORTFOLIO TRACKER
        if t.startswith('/beli'):
            parts = text.strip().split()
            if len(parts) >= 3:
                try:
                    buy_price = float(parts[1].replace(',', '').replace('.', ''))
                    amount = float(parts[2].replace(',', '').replace('.', ''))
                    entry = {
                        "buy_price": buy_price,
                        "amount": amount,
                        "date": datetime.now().strftime('%d/%m/%Y %H:%M'),
                        "total": buy_price * amount
                    }
                    self.portfolio.append(entry)
                    self.total_invested += entry['total']
                    return (
                        f"✅ *PEMBELIAN DICATAT*\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"💰 Harga beli: Rp {buy_price:,.0f}\n"
                        f"📦 Jumlah: {amount:,.2f} ISLM\n"
                        f"💵 Total: Rp {entry['total']:,.0f}\n"
                        f"📅 Tanggal: {entry['date']}\n\n"
                        f"Ketik /portfolio untuk lihat P&L"
                    )
                except ValueError:
                    return "❌ Format salah.\n*Contoh:* /beli 150 1000\n(beli 1000 ISLM @Rp 150)"
            return "💰 *Cara pakai:*\n/beli [harga] [jumlah]\n*Contoh:* /beli 150 1000"

        if t.startswith('/jual'):
            parts = text.strip().split()
            if len(parts) >= 2:
                try:
                    idx = int(parts[1]) - 1
                    if 0 <= idx < len(self.portfolio):
                        removed = self.portfolio.pop(idx)
                        sell_price = c['price']
                        pnl = (sell_price - removed['buy_price']) * removed['amount']
                        pnl_pct = ((sell_price / removed['buy_price']) - 1) * 100
                        emoji = "🟢" if pnl >= 0 else "🔴"
                        return (
                            f"{emoji} *POSISI DITUTUP*\n"
                            f"Beli: Rp {removed['buy_price']:,.0f} → Jual: Rp {sell_price:,.0f}\n"
                            f"P&L: Rp {pnl:,.0f} ({pnl_pct:+.1f}%)"
                        )
                    return "❌ Nomor posisi tidak valid."
                except ValueError:
                    pass
            return "💰 *Cara pakai:* /jual [nomor posisi]\nLihat nomor di /portfolio"

        if t in ['/portfolio', '/porto', 'portfolio', 'porto']:
            if not self.portfolio:
                return "💼 *PORTFOLIO KOSONG*\nBelum ada posisi.\nGunakan /beli [harga] [jumlah] untuk catat pembelian."
            
            current_price = c['price']
            lines = ["💼 *PORTFOLIO ISLM*", "━━━━━━━━━━━━━━━━━━━━━━"]
            total_value = 0
            total_cost = 0
            
            for i, pos in enumerate(self.portfolio, 1):
                value = current_price * pos['amount']
                cost = pos['buy_price'] * pos['amount']
                pnl = value - cost
                pnl_pct = ((current_price / pos['buy_price']) - 1) * 100 if pos['buy_price'] > 0 else 0
                emoji = "🟢" if pnl >= 0 else "🔴"
                total_value += value
                total_cost += cost
                lines.append(
                    f"\n*#{i}* | {pos['date']}\n"
                    f"  Beli: Rp {pos['buy_price']:,.0f} x {pos['amount']:,.2f}\n"
                    f"  {emoji} P&L: Rp {pnl:,.0f} ({pnl_pct:+.1f}%)"
                )
            
            total_pnl = total_value - total_cost
            total_pnl_pct = ((total_value / total_cost) - 1) * 100 if total_cost > 0 else 0
            emoji_total = "🟢" if total_pnl >= 0 else "🔴"
            
            lines += [
                f"\n━━━━━━━━━━━━━━━━━━━━━━",
                f"💰 *Harga sekarang:* Rp {current_price:,.0f}",
                f"💵 *Total invest:* Rp {total_cost:,.0f}",
                f"💰 *Nilai sekarang:* Rp {total_value:,.0f}",
                f"{emoji_total} *Total P&L:* Rp {total_pnl:,.0f} ({total_pnl_pct:+.1f}%)",
                f"\n💡 _/jual [nomor] untuk tutup posisi_"
            ]
            return "\n".join(lines)

        # V7: CUSTOM PRICE TARGET
        if t.startswith('/target'):
            parts = text.strip().split()
            if len(parts) >= 2:
                if parts[1] == 'hapus' or parts[1] == 'clear':
                    self.price_targets.clear()
                    return "✅ Semua target dihapus."
                try:
                    target_price = float(parts[1].replace(',', '').replace('.', ''))
                    direction = "above" if target_price > c['price'] else "below"
                    self.price_targets.append({
                        "price": target_price,
                        "direction": direction,
                        "notified": False
                    })
                    arrow = "⬆️" if direction == "above" else "⬇️"
                    return (
                        f"🎯 *TARGET ALERT SET*\n"
                        f"{arrow} Notif saat ISLM {'naik ke' if direction == 'above' else 'turun ke'} Rp {target_price:,.0f}\n"
                        f"💰 Harga sekarang: Rp {c['price']:,.0f}\n\n"
                        f"Total target aktif: {len(self.price_targets)}"
                    )
                except ValueError:
                    pass
            # Show current targets
            if self.price_targets:
                lines = ["🎯 *TARGET ALERTS*", "━━━━━━━━━━━━━━━━━━━━━━"]
                for i, tgt in enumerate(self.price_targets, 1):
                    arrow = "⬆️" if tgt['direction'] == 'above' else "⬇️"
                    status = "✅" if tgt['notified'] else "⏳"
                    lines.append(f"{status} #{i} {arrow} Rp {tgt['price']:,.0f}")
                lines.append(f"\n💰 Sekarang: Rp {c['price']:,.0f}")
                lines.append("\n_/target hapus — hapus semua_")
                return "\n".join(lines)
            return "🎯 *Cara pakai:*\n/target 200 — Notif saat ISLM Rp 200\n/target hapus — Hapus semua"

        # V7: BOT STATS
        if t in ['/stats', 'stats', '/statistik']:
            uptime = time.time() - self.start_time
            days = int(uptime // 86400)
            hours = int((uptime % 86400) // 3600)
            mins = int((uptime % 3600) // 60)
            uptime_str = f"{days}d {hours}h {mins}m" if days else f"{hours}h {mins}m"
            
            return (
                f"📊 *BOT STATISTICS*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"⏱️ Uptime: {uptime_str}\n"
                f"💬 Pesan dijawab: {self._stats['messages_handled']}\n"
                f"🧠 AI calls: {self._stats['ai_calls']}\n"
                f"🚨 Alerts terkirim: {self._stats['alerts_sent']}\n"
                f"🖼️ Foto dianalisa: {self._stats['photos_analyzed']}\n"
                f"⚠️ Errors: {self._stats['errors']}\n\n"
                f"🧠 *Groq Usage:*\n"
                f"  RPM: {len(self._groq_requests)}/{self._groq_rpm_limit}\n"
                f"  Daily: {self._groq_daily_count}/14400\n\n"
                f"💼 Portfolio: {len(self.portfolio)} posisi\n"
                f"🎯 Targets: {len(self.price_targets)} aktif\n"
                f"⚡ Mode: {'VOLATILE' if self.volatility_mode else 'Normal'}\n"
                f"📈 Memory: {len(self._chat_memory)} msgs"
            )

        # V7: NOTIFICATION CONTROL
        if t in ['/pause', 'pause', '/diam', 'diam']:
            self.notifications_paused = True
            return (
                "⏸️ *NOTIFIKASI DI-PAUSE*\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "Bot tidak akan kirim update otomatis.\n"
                "Tapi kamu tetap bisa chat dan tanya AI!\n\n"
                "Ketik /resume untuk aktifkan kembali."
            )

        if t in ['/resume', 'resume', '/aktif', 'aktif', '/nyala']:
            self.notifications_paused = False
            return (
                "▶️ *NOTIFIKASI AKTIF KEMBALI*\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                "Bot akan kirim update otomatis lagi:\n"
                "  ⚡ 30m / 📊 1h / 📅 Daily\n"
                "  🚨 Price Alerts / ⚡ Volatile Mode\n\n"
                "Ketik /pause untuk matikan lagi."
            )

        # STATUS
        if any(k in t for k in ['/status', 'status', 'harga', 'price', 'berapa', 'market', 'kondisi']):
            bb_info = f"BB: Rp {c['bb_lower']:,.0f} — Rp {c['bb_upper']:,.0f}" if c['bb_upper'] else "BB: N/A"
            resp = (
                f"📊 *STATUS ISLM*\n\n"
                f"💰 Harga: Rp {c['price']:,.0f}\n"
                f"📢 Sinyal: {sig['label']} ({sig['confidence']*100:.0f}%)\n"
                f"📈 Trend: {sig['trend']}\n"
                f"🏷️ Fase: {c['market_phase']}\n\n"
                f"📊 RSI: {c['rsi']:.1f} | MACD: {c['macd_val']:.2f}\n"
                f"📐 {bb_info}\n"
                f"🐋 {c['whale_label']}\n"
            )
            if ml.get('ml_available'):
                resp += f"🤖 ML: {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)\n"
            resp += f"\n⏰ {c['last_update']} WIB"
            return resp

        # PREDICT
        if any(k in t for k in ['/predict', 'prediksi', 'ramal', 'forecast', 'target', 'naik', 'turun']):
            lines = [f"🔮 *PREDIKSI AI ISLM*\n💰 Harga: Rp {c['price']:,.0f}\n"]
            for k, p in c["predictions"].items():
                lines.append(
                    f"*{p['label']}:* Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%) {p['direction']}\n"
                    f"  Range: Rp {p['low']:,.0f} — Rp {p['high']:,.0f}\n"
                    f"  Confidence: {p['confidence']:.0f}%"
                )
            if c['supports']:
                lines.append(f"\n🟢 Support: {', '.join(f'Rp {s:,.0f}' for s in c['supports'])}")
            if c['resistances']:
                lines.append(f"🔴 Resistance: {', '.join(f'Rp {r:,.0f}' for r in c['resistances'])}")
            lines.append(f"\n📢 Sinyal: {sig['label']}")
            return "\n".join(lines)

        # TECHNICAL ANALYSIS
        if any(k in t for k in ['/analisa', 'analisa', 'teknikal', 'technical', 'indikator']):
            lines = [f"📊 *ANALISA TEKNIKAL ISLM V4*\n"]
            lines.append(f"📈 RSI: {c['rsi']:.1f}")
            lines.append(f"📉 MACD: {c['macd_val']:.2f} | Hist: {c['hist']:+.2f}")
            lines.append(f"📊 Stoch: {c['stoch_k']:.1f}")
            if c['atr']: lines.append(f"📏 ATR: {c['atr']:.2f}")
            if c['bb_upper']:
                lines.append(f"📐 BB: Rp {c['bb_lower']:,.0f} — Rp {c['bb_upper']:,.0f}")
            # ProTA extras
            if pta.get('adx'): lines.append(f"💪 ADX: {pta['adx']:.0f} ({'Trend Kuat' if pta['adx'] > 25 else 'Sideways'})")
            if pta.get('ema_9') and pta.get('ema_21'):
                cross = "Golden Cross ✨" if pta['ema_9'] > pta['ema_21'] else "Death Cross 💀"
                lines.append(f"📊 EMA: {cross}")
            if pta.get('mfi'): lines.append(f"💰 MFI: {pta['mfi']:.0f}")
            if pta.get('williams_r') is not None: lines.append(f"📉 Williams %R: {pta['williams_r']:.0f}")
            if pta.get('roc') is not None: lines.append(f"📈 ROC: {pta['roc']:.2f}")
            if pta.get('obv'): lines.append(f"📊 OBV: {pta['obv']:,.0f}")
            if pta.get('vwap'): lines.append(f"📊 VWAP: Rp {pta['vwap']:,.0f}")
            lines.append(f"\n🏷️ Fase: {c['market_phase']}")
            lines.append(f"📢 Sinyal: {sig['label']}")
            return "\n".join(lines)

        # ML SIGNAL
        if any(k in t for k in ['/ml', 'machine learning', 'ai model', 'gradient']):
            if ml.get('ml_available'):
                lines = [
                    f"🤖 *ML SIGNAL CLASSIFIER*\n",
                    f"📢 Prediksi ML: {ml['ml_signal']}",
                    f"🎯 Confidence: {ml['ml_confidence']*100:.0f}%\n",
                ]
                if ml.get('ml_class_probs'):
                    lines.append("📊 *Probabilitas:*")
                    for cls, prob in ml['ml_class_probs'].items():
                        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                        lines.append(f"  {cls}: {bar} {prob*100:.0f}%")
                lines.append(f"\n📢 Rule-Based: {sig['label']}")
                return "\n".join(lines)
            return "🤖 ML belum cukup data. Perlu minimal 60 candles."

        # NEWS
        if any(k in t for k in ['/news', 'berita', 'news', 'fundamental', 'kabar', 'sentimen']):
            news = NewsEngine.generate_news(c['market_phase'], c['whale_ratio'])
            return (
                f"📰 *BERITA & FUNDAMENTAL ISLM*\n\n"
                f"{c['f_news']}\n\n"
                f"📰 {news}\n\n"
                f"📊 Skor: {c['f_score']}/10 | Fase: {c['market_phase']}\n"
                f"📢 Sinyal: {sig['label']}"
            )

        # WHALE
        if any(k in t for k in ['/whale', 'whale', 'paus', 'order book']):
            d = ""
            if c['whale_ratio'] > 0.6: d = "⚡ Whale sedang AKUMULASI!"
            elif c['whale_ratio'] < 0.4: d = "⚠️ Whale sedang DISTRIBUSI!"
            else: d = "⚖️ Whale seimbang."
            return f"🐋 *WHALE TRACKER*\n\n📊 Ratio: {c['whale_ratio']*100:.0f}%\n📢 {c['whale_label']}\n\n{d}"

        # SIGNAL
        if any(k in t for k in ['/sinyal', 'sinyal', 'signal', 'beli', 'jual', 'buy', 'sell']):
            lines = [f"📢 *SINYAL AI V4: {sig['label']}*\n"]
            lines.append(f"🎯 Confidence: {sig['confidence']*100:.0f}%")
            lines.append(f"📈 Trend: {sig['trend']}\n")
            if ml.get('ml_available'):
                lines.append(f"🤖 ML: {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)\n")
            lines.append("🧠 *Reasoning:*")
            for r in sig.get("reasons", []):
                lines.append(f"• {r}")
            if c['candle_patterns']:
                lines.append(f"\n🕯️ Pola: {', '.join(c['candle_patterns'])}")
            return "\n".join(lines)

        # SUPPORT/RESISTANCE
        if any(k in t for k in ['/level', 'support', 'resistance', 'level', 'sr']):
            lines = [f"📐 *SUPPORT & RESISTANCE*\n💰 Harga: Rp {c['price']:,.0f}\n"]
            if c['supports']:
                for s in c['supports']:
                    dist = ((c['price'] - s) / (c['price'] + 1e-10)) * 100
                    lines.append(f"🟢 Support: Rp {s:,.0f} ({dist:+.1f}% dari harga)")
            else:
                lines.append("🟢 Support: Belum cukup data")
            if c['resistances']:
                for r in c['resistances']:
                    dist = ((r - c['price']) / (c['price'] + 1e-10)) * 100
                    lines.append(f"🔴 Resistance: Rp {r:,.0f} (+{dist:.1f}% dari harga)")
            else:
                lines.append("🔴 Resistance: Belum cukup data")
            return "\n".join(lines)

        # COIN — Multi-Coin Check
        if any(k in t for k in ['/coin', '/harga']):
            parts = text.strip().split()
            if len(parts) >= 2:
                coin_name = parts[1].lower().replace('/', '')
                pair = f"{coin_name}idr"
                try:
                    p = self.api.get_price(pair)
                    if p.get('success'):
                        return (
                            f"💰 *{coin_name.upper()}/IDR*\n\n"
                            f"💵 Harga: Rp {p['last']:,.0f}\n"
                            f"📈 High 24h: Rp {p['high']:,.0f}\n"
                            f"📉 Low 24h: Rp {p['low']:,.0f}\n"
                            f"📊 Volume: {p['vol']:,.2f}\n\n"
                            f"_Ketik nama coin untuk analisa AI:_\n"
                            f"_Contoh: \"analisa {coin_name.upper()} dong\"_"
                        )
                    else:
                        return f"❌ Coin '{coin_name.upper()}' tidak ditemukan di Indodax."
                except:
                    return f"❌ Gagal ambil data {coin_name.upper()}"
            else:
                # Show top coins
                try:
                    prices = self.api.get_multi_price(['islmidr', 'btcidr', 'ethidr', 'xrpidr', 'dogeidr'])
                    lines = ["💰 *TOP COINS INDODAX*\n"]
                    names = {'islmidr': 'ISLM ⭐', 'btcidr': 'BTC', 'ethidr': 'ETH', 'xrpidr': 'XRP', 'dogeidr': 'DOGE'}
                    for pair_id, name in names.items():
                        if pair_id in prices:
                            p = prices[pair_id]
                            lines.append(f"  {name}: Rp {p['last']:,.0f}")
                    lines.append("\n_Ketik_ `/coin btc` _untuk detail._")
                    return "\n".join(lines)
                except:
                    return "❌ Gagal ambil data multi-coin"

        # RISK METRICS
        if any(k in t for k in ['/risk', 'risiko', 'sharpe', 'drawdown']):
            if len(self.price_history) >= 10:
                risk = RiskMetrics.full_report(self.price_history)
                return (
                    f"📊 *RISK METRICS ISLM*\n\n"
                    f"📈 Sharpe Ratio: {risk['sharpe']:.2f}\n"
                    f"📉 Max Drawdown: {risk['max_dd']:.1f}%\n"
                    f"✅ Win Rate: {risk['win_rate']:.0f}%\n"
                    f"⚠️ VaR (95%): {risk['var_95']:.2f}%\n\n"
                    f"{'🟢 Risiko RENDAH' if risk['max_dd'] < 5 else '🟡 Risiko SEDANG' if risk['max_dd'] < 15 else '🔴 Risiko TINGGI'}"
                )
            return "⏳ Belum cukup data untuk risk metrics (min 10 data point)"

        # SECURITY — V5 Enhanced
        if any(k in t for k in ['/security', 'keamanan', 'aman', 'security']):
            uptime = (time.time() - self.start_time) / 3600
            sec_report = self.security.get_full_report()
            return (
                f"{sec_report}\n\n"
                f"✅ Bot V5: *Aktif*\n"
                f"✅ ML Engine: *{'Aktif' if ml.get('ml_available') else 'Loading'}*\n"
                f"✅ ProTA: *Aktif* | V5 Modules: *Aktif*\n"
                f"🔑 API Keys: *Env Variables*\n"
                f"📡 Auto: 30m + 1h + Daily\n"
                f"⏱️ Uptime: {uptime:.1f} jam"
            )

        # CANDLE
        if any(k in t for k in ['/candle', 'pola', 'pattern', 'candle']):
            if c['candle_patterns']:
                return f"🕯️ *POLA CANDLESTICK:*\n\n" + "\n".join(f"• {p}" for p in c['candle_patterns'])
            return "🕯️ Tidak ada pola terdeteksi."

        # DEFAULT → GROQ AI (Free Llama 3.3 70B)
        return self._ask_groq_ai(text)

    # ============================================
    # GROQ AI V6 — Genius Mode + Memory + Chain-of-Thought
    # ============================================
    def _ask_groq_ai(self, user_message):
        """Send user message to Groq AI with full market context, memory, and genius-level understanding."""
        api_key = Config.GROQ_API_KEY if hasattr(Config, 'GROQ_API_KEY') else ''
        if not api_key:
            api_key = os.environ.get('GROQ_API_KEY', '')
        if not api_key:
            return (
                "🤖 Coba ketik:\n"
                "/status — Market\n/predict — Prediksi\n/analisa — Teknikal\n"
                "/ml — Sinyal ML\n/sinyal — Sinyal AI\n/level — Support/Resistance\n"
                "/whale — Whale\n/news — Berita\n/candle — Pola"
            )
        try:
            from groq import Groq
            from backend.core_logic import BTCCorrelation, RiskMetrics
            client = Groq(api_key=api_key)
            c = self._cache
            sig = c["ai_signal"]
            ml = c["ml_result"]

            # Build rich V5 market context
            context_lines = [
                f"=== DATA ISLM REAL-TIME ===",
                f"HARGA ISLM: Rp {c['price']:,.0f}",
                f"SINYAL AI: {sig['label']} (Confidence: {sig['confidence']*100:.0f}%)",
                f"TREND: {sig['trend']} | FASE: {c['market_phase']}",
                f"RSI: {c['rsi']:.1f} | MACD: {c['macd_val']:.2f} | MACD Hist: {c['hist']:+.2f}",
                f"Stoch: {c['stoch_k']:.1f}",
                f"WHALE: {c['whale_label']} ({c['whale_ratio']*100:.0f}% buy)",
            ]
            if c['bb_upper']:
                context_lines.append(f"BB: Rp {c['bb_lower']:,.0f} — Rp {c['bb_upper']:,.0f}")
            if c['atr']:
                context_lines.append(f"ATR: {c['atr']:.2f}")

            pta = c.get('pro_ta', {})
            extras = []
            if pta.get('adx'): extras.append(f"ADX={pta['adx']:.0f}")
            if pta.get('mfi'): extras.append(f"MFI={pta['mfi']:.0f}")
            if pta.get('ema_9') and pta.get('ema_21'):
                extras.append(f"EMA9{'>' if pta['ema_9'] > pta['ema_21'] else '<'}EMA21")
            if extras: context_lines.append(f"INDIKATOR: {' | '.join(extras)}")

            if ml.get('ml_available'):
                context_lines.append(f"ML SIGNAL: {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)")
            if c['supports']:
                context_lines.append(f"SUPPORT: {', '.join(f'Rp {s:,.0f}' for s in c['supports'])}")
            if c['resistances']:
                context_lines.append(f"RESISTANCE: {', '.join(f'Rp {r:,.0f}' for r in c['resistances'])}")

            for k, p in c["predictions"].items():
                context_lines.append(f"PREDIKSI {p['label']}: Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%) {p['direction']}")

            # V5: BTC Correlation
            try:
                btc_price = BTCCorrelation.fetch_btc_price()
                if btc_price > 0:
                    context_lines.append(f"\n=== CROSS-MARKET ===")
                    context_lines.append(f"BTC/IDR: Rp {btc_price:,.0f}")
            except: pass

            # V5: Risk Metrics
            try:
                if len(self.price_history) >= 10:
                    risk = RiskMetrics.full_report(self.price_history)
                    context_lines.append(f"\n=== RISK METRICS ===")
                    context_lines.append(f"Sharpe Ratio: {risk['sharpe']:.2f}")
                    context_lines.append(f"Max Drawdown: {risk['max_dd']:.1f}%")
                    context_lines.append(f"Win Rate: {risk['win_rate']:.0f}%")
                    context_lines.append(f"VaR (95%): {risk['var_95']:.2f}%")
            except: pass

            # V5: Multi-coin check (if user asks about other coins)
            msg_lower = user_message.lower()
            other_coins = {'btc': 'btcidr', 'eth': 'ethidr', 'sol': 'solidr',
                          'xrp': 'xrpidr', 'doge': 'dogeidr', 'ada': 'adaidr',
                          'dot': 'dotidr', 'bnb': 'bnbidr', 'link': 'linkidr'}
            for coin_name, pair in other_coins.items():
                if coin_name in msg_lower:
                    try:
                        p = self.api.get_price(pair)
                        if p.get('success'):
                            context_lines.append(f"\n=== {coin_name.upper()} ===")
                            context_lines.append(f"{coin_name.upper()}/IDR: Rp {p['last']:,.0f} | H: Rp {p['high']:,.0f} | L: Rp {p['low']:,.0f}")
                    except: pass

            context_lines.append("\nREASONING AI:")
            for r in sig.get("reasons", []):
                context_lines.append(f"  - {r}")

            market_context = "\n".join(context_lines)

            # === V6: GENIUS SYSTEM PROMPT ===
            system_prompt = (
                "Kamu adalah ISLM AI — asisten trading super pintar yang bisa memahami KONTEKS dan MAKSUD user "
                "dengan sangat baik, bahkan jika pertanyaan mereka ambigu, singkat, atau menggunakan bahasa gaul.\n\n"

                "=== CARA BERPIKIR (CHAIN-OF-THOUGHT) ===\n"
                "Sebelum menjawab, SELALU lakukan ini di dalam pikiranmu:\n"
                "1. PAHAMI MAKSUD: Apa yang sebenarnya user tanyakan? Jangan hanya baca literal.\n"
                "   - 'gimana?' → user tanya kondisi market ISLM\n"
                "   - 'jadi beli ga?' → user minta rekomendasi aksi\n"
                "   - 'kok turun sih' → user frustasi + minta penjelasan\n"
                "   - 'worth it ga?' → user minta analisa risk/reward\n"
                "   - 'kapan naik?' → user minta prediksi timing\n"
                "   - 'lagi galau nih' → user butuh dukungan emosional + saran\n"
                "   - 'bagus ga ISLM?' → user minta opini + data fundamental\n"
                "   - 'aman ga hold?' → user minta analisa risk untuk hold position\n"
                "2. TENTUKAN TIPE: Apakah ini pertanyaan analitis, emosional, komparatif, atau edukasi?\n"
                "3. PILIH GAYA: Sesuaikan gaya jawaban dengan tipe pertanyaan.\n"
                "4. JAWAB DENGAN DATA: Selalu backup opini dengan data real-time.\n\n"

                "=== KEPRIBADIAN ===\n"
                "- Kamu sahabat trading yang SANGAT pintar dan pengertian\n"
                "- Kamu paham bahasa Indonesia gaul, slang, typo, singkatan\n"
                "  (gw=saya, lu=kamu, gak/ga/kagak=tidak, gimana/gmn=bagaimana, bgt=banget, "
                "   dah/udh=sudah, bisa/bs=bisa, yg=yang, emg/emang=memang, dll)\n"
                "- Kalau user curhat/frustasi → DENGARKAN dulu, empati, baru kasih solusi\n"
                "- Kalau user senang → ikut senang, tapi ingatkan risk management\n"
                "- Kalau pertanyaan ambigu → tebak maksud terbaik, jawab, lalu tanyakan\n"
                "- JANGAN pernah jawab 'saya tidak mengerti' — selalu coba interpretasi\n"
                "- Jawab FOKUS dan RELEVAN, jangan bertele-tele\n\n"

                "=== GAYA JAWABAN PER TIPE ===\n"
                "📊 ANALITIS (harga, prediksi, teknikal):\n"
                "  → Langsung kasih data + angka + rekomendasi aksi\n"
                "  → Format: Kondisi → Analisa → Aksi yang disarankan\n\n"
                "💭 EMOSIONAL (curhat, galau, takut, senang):\n"
                "  → Empati dulu → Validasi perasaan → Saran praktis\n"
                "  → Gunakan bahasa hangat dan supportif\n\n"
                "⚖️ KOMPARATIF (bandingkan coin, strategi):\n"
                "  → Tabel perbandingan → Pro/Kontra → Rekomendasi\n\n"
                "📚 EDUKASI (apa itu RSI, cara trading, dll):\n"
                "  → Jelaskan sederhana → Contoh nyata dengan data ISLM\n\n"

                "=== KEMAMPUAN ===\n"
                "- Analisis teknikal mendalam (RSI, MACD, BB, Fibonacci, S/R, dll)\n"
                "- Prediksi harga berdasarkan data (Monte Carlo, ML)\n"
                "- Risk metrics (Sharpe, MaxDD, WinRate, VaR)\n"
                "- Multi-coin analysis (BTC, ETH, SOL, dll di Indodax)\n"
                "- Edukasi trading dan crypto\n"
                "- Emotional support dan motivasi\n"
                "- Menjawab follow-up questions dengan konteks percakapan sebelumnya\n\n"

                f"=== DATA REAL-TIME ===\n{market_context}\n\n"

                "=== ATURAN WAJIB (SANGAT PENTING!) ===\n"
                "- JAWAB SINGKAT DAN PADAT! Maksimal 3-5 paragraf pendek.\n"
                "- JANGAN panjang lebar. User BENCI jawaban yang terlalu panjang.\n"
                "- Sertakan angka/data real-time yang RELEVAN saja, jangan semua data.\n"
                "- Gunakan emoji SECUKUPNYA (max 5-8 emoji per jawaban)\n"
                "- Format rapi: bullet points singkat, bold untuk angka penting\n"
                "- Bahasa Indonesia natural dan santai\n"
                "- Akhiri dengan 1 kalimat actionable insight\n"
                "- Jika user tanya di luar crypto, jawab singkat lalu kaitkan ke investasi\n"
                "- INGAT: SINGKAT = BAGUS. Panjang = SPAM. User sudah komplain soal ini!"
            )

            # === V7: RATE LIMITER ===
            now_ts = time.time()
            # Clean old requests (older than 60s)
            self._groq_requests = [t for t in self._groq_requests if now_ts - t < 60]
            if len(self._groq_requests) >= self._groq_rpm_limit:
                wait_time = int(60 - (now_ts - self._groq_requests[0]))
                return f"⏳ AI istirahat {wait_time} detik (limit 30 req/menit).\nCoba lagi sebentar ya! 🙏"

            # Daily counter reset
            if now_ts - self._groq_daily_reset >= 86400:
                self._groq_daily_count = 0
                self._groq_daily_reset = now_ts

            # === V6: BUILD MESSAGES WITH MEMORY ===
            messages = [{"role": "system", "content": system_prompt}]

            # Add conversation memory (last messages for context)
            if hasattr(self, '_chat_memory') and self._chat_memory:
                for mem in self._chat_memory[-6:]:  # Last 6 exchanges (3 pairs)
                    messages.append(mem)

            messages.append({"role": "user", "content": user_message})

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=400,
                temperature=0.7,
            )
            self._groq_requests.append(time.time())
            self._groq_daily_count += 1
            ai_reply = response.choices[0].message.content

            # Save to memory
            if not hasattr(self, '_chat_memory'):
                self._chat_memory = []
            self._chat_memory.append({"role": "user", "content": user_message})
            self._chat_memory.append({"role": "assistant", "content": ai_reply})
            # Keep only last 10 messages
            if len(self._chat_memory) > 10:
                self._chat_memory = self._chat_memory[-10:]

            return ai_reply
        except Exception as e:
            safe_print(f"[Groq Error] {e}")
            return (
                f"🤖 AI sedang sibuk. Gunakan perintah:\n"
                "/status — Market\n/predict — Prediksi\n/sinyal — Sinyal AI"
            )

    # ============================================
    # V7: IMAGE ANALYSIS (Groq Vision)
    # ============================================
    def handle_photo(self, photo_list, caption=''):
        """Download and analyze an image sent by user."""
        try:
            # Get highest resolution photo
            photo = photo_list[-1]  # Last = highest res
            file_id = photo['file_id']

            # Get file path from Telegram
            file_info = requests.get(
                f"{self.base_url}/getFile",
                params={'file_id': file_id}, timeout=10
            ).json()

            if not file_info.get('ok'):
                return "❌ Gagal mengambil gambar dari Telegram."

            file_path = file_info['result']['file_path']
            file_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"

            # Download image
            img_resp = requests.get(file_url, timeout=30)
            if img_resp.status_code != 200:
                return "❌ Gagal download gambar."

            import base64
            img_b64 = base64.b64encode(img_resp.content).decode('utf-8')

            # Determine image type from file path
            ext = file_path.split('.')[-1].lower() if '.' in file_path else 'jpg'
            mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"

            # Send to Groq Vision
            api_key = Config.GROQ_API_KEY or os.environ.get('GROQ_API_KEY', '')
            if not api_key:
                return "❌ AI Vision tidak tersedia (API key missing)."

            from groq import Groq
            client = Groq(api_key=api_key)

            # Rate limit check
            now_ts = time.time()
            self._groq_requests = [t for t in self._groq_requests if now_ts - t < 60]
            if len(self._groq_requests) >= self._groq_rpm_limit:
                wait_time = int(60 - (now_ts - self._groq_requests[0]))
                return f"⏳ AI Vision istirahat {wait_time} detik. Coba lagi ya!"

            user_prompt = caption if caption else "Analisa gambar ini secara detail. Jelaskan apa yang kamu lihat."

            response = client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {"role": "system", "content":
                        "Kamu AI analyst yang bisa menganalisa gambar. "
                        "Kemampuanmu: analisa chart/grafik trading, soal matematika, "
                        "pengenalan objek, pengetahuan umum, membaca teks di gambar. "
                        "Jawab dalam Bahasa Indonesia yang santai tapi informatif. "
                        "JAWAB SINGKAT DAN PADAT, maksimal 3-4 paragraf."
                    },
                    {"role": "user", "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {
                            "url": f"data:{mime};base64,{img_b64}"
                        }}
                    ]}
                ],
                max_tokens=400,
                temperature=0.7,
            )
            self._groq_requests.append(time.time())
            self._groq_daily_count += 1

            result = response.choices[0].message.content
            return f"🖼️ *ANALISA GAMBAR*\n━━━━━━━━━━━━━━━━━━━━━━\n\n{result}"

        except Exception as e:
            safe_print(f"[VISION Error] {e}")
            return f"❌ Gagal analisa gambar: {str(e)[:100]}"

    # ============================================
    # V7: PHONE NUMBER LOOKUP
    # ============================================
    def lookup_phone(self, number):
        """Lookup phone number info using Numverify API."""
        try:
            numverify_key = os.environ.get('NUMVERIFY_API_KEY', '')

            # Clean the number
            clean = number.strip().replace(' ', '').replace('-', '').replace('(', '').replace(')', '')
            if clean.startswith('0'):
                clean = '62' + clean[1:]  # Convert 08xx to 628xx
            if not clean.startswith('+'):
                clean = '+' + clean if not clean.startswith('62') else '+' + clean

            if numverify_key:
                # Use Numverify API
                resp = requests.get(
                    'http://apilayer.net/api/validate',
                    params={
                        'access_key': numverify_key,
                        'number': clean,
                        'country_code': 'ID',
                        'format': 1
                    },
                    timeout=10
                )
                data = resp.json()

                if data.get('valid'):
                    lines = [
                        f"📱 *HASIL CEK NOMOR*",
                        f"━━━━━━━━━━━━━━━━━━━━━━",
                        f"📞 Nomor: {data.get('international_format', clean)}",
                        f"✅ Status: *VALID*",
                        f"🏢 Operator: {data.get('carrier', 'Unknown')}",
                        f"📍 Lokasi: {data.get('location', 'Unknown')}",
                        f"📡 Tipe: {data.get('line_type', 'Unknown')}",
                        f"🌍 Negara: {data.get('country_name', 'Unknown')}",
                    ]
                    return "\n".join(lines)
                else:
                    return f"❌ Nomor *{clean}* tidak valid atau tidak ditemukan."
            else:
                # Fallback: basic Indonesia number analysis
                lines = [
                    f"📱 *HASIL CEK NOMOR*",
                    f"━━━━━━━━━━━━━━━━━━━━━━",
                    f"📞 Nomor: {clean}",
                ]

                # Detect Indonesian operator by prefix
                prefix = clean.replace('+62', '0')[:4]
                operators = {
                    '0811': 'Telkomsel (Halo)', '0812': 'Telkomsel (Simpati)',
                    '0813': 'Telkomsel (Simpati)', '0821': 'Telkomsel (Simpati)',
                    '0822': 'Telkomsel (Simpati)', '0823': 'Telkomsel (As)',
                    '0851': 'Telkomsel (As)', '0852': 'Telkomsel (As)',
                    '0853': 'Telkomsel (As)',
                    '0814': 'Indosat (IM3)', '0815': 'Indosat (Matrix)',
                    '0816': 'Indosat (Mentari)', '0855': 'Indosat (Matrix)',
                    '0856': 'Indosat (IM3)', '0857': 'Indosat (IM3)',
                    '0858': 'Indosat (Mentari)',
                    '0817': 'XL Axiata', '0818': 'XL Axiata',
                    '0819': 'XL Axiata', '0859': 'XL Axiata',
                    '0877': 'XL Axiata', '0878': 'XL Axiata',
                    '0831': 'Axis', '0832': 'Axis',
                    '0833': 'Axis', '0838': 'Axis',
                    '0895': 'Three (3)', '0896': 'Three (3)',
                    '0897': 'Three (3)', '0898': 'Three (3)',
                    '0899': 'Three (3)',
                    '0881': 'Smartfren', '0882': 'Smartfren',
                    '0883': 'Smartfren', '0884': 'Smartfren',
                    '0885': 'Smartfren', '0886': 'Smartfren',
                    '0887': 'Smartfren', '0888': 'Smartfren',
                    '0889': 'Smartfren',
                }

                op = operators.get(prefix, None)
                if not op:
                    # Try 3-digit prefix
                    prefix3 = clean.replace('+62', '0')[:3]
                    for k, v in operators.items():
                        if k.startswith(prefix3):
                            op = v
                            break

                if op:
                    lines.append(f"✅ Status: *VALID (Indonesia)*")
                    lines.append(f"🏢 Operator: *{op}*")
                    lines.append(f"📍 Negara: Indonesia 🇮🇩")
                    lines.append(f"📡 Tipe: Mobile")
                else:
                    lines.append(f"⚠️ Operator tidak terdeteksi")
                    lines.append(f"📍 Cek prefix: {prefix}")

                lines.append(f"\n💡 _Tip: Set NUMVERIFY\\_API\\_KEY untuk info lebih lengkap (gratis di numverify.com)_")
                return "\n".join(lines)

        except Exception as e:
            safe_print(f"[PHONE Error] {e}")
            return f"❌ Gagal cek nomor: {str(e)[:100]}"

    # ============================================
    # V6: SMART PRICE ALERT
    # ============================================
    def check_price_alert(self):
        """Check if price moved significantly and send alert."""
        now = time.time()
        price = self._cache.get('price', 0)
        if not price or not self.last_alert_price:
            self.last_alert_price = price
            return

        # Reset hourly counter
        if now - self.alerts_hour_reset >= 3600:
            self.alerts_this_hour = 0
            self.alerts_hour_reset = now

        # Anti-spam checks
        if now - self.last_alert_time < self.alert_cooldown:
            return
        if self.alerts_this_hour >= self.max_alerts_per_hour:
            return

        # Calculate change
        change_pct = ((price - self.last_alert_price) / (self.last_alert_price + 1e-10)) * 100

        if abs(change_pct) >= self.alert_threshold * 100:  # ±3%
            sig = self._cache.get('ai_signal', {})
            direction = "📈 NAIK" if change_pct > 0 else "📉 TURUN"
            emoji = "🚀" if change_pct > 5 else "📈" if change_pct > 0 else "🔻" if change_pct < -5 else "📉"

            lines = [
                f"{emoji} *PRICE ALERT!*",
                f"━━━━━━━━━━━━━━━━━━━━━━",
                f"💰 *ISLM {direction} {abs(change_pct):.1f}%*",
                f"",
                f"📊 Sebelum: Rp {self.last_alert_price:,.0f}",
                f"📊 Sekarang: Rp {price:,.0f}",
                f"📢 Sinyal: {sig.get('label', 'N/A')}",
                f"📈 Trend: {sig.get('trend', 'N/A')}",
            ]

            # Add AI quick comment
            try:
                api_key = Config.GROQ_API_KEY or os.environ.get('GROQ_API_KEY', '')
                if api_key:
                    from groq import Groq
                    client = Groq(api_key=api_key)
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content":
                                "Kamu AI trading. ISLM baru saja bergerak signifikan. "
                                "Beri komentar singkat 2 kalimat: (1) kenapa ini terjadi, "
                                "(2) apa yang harus dilakukan trader. Bahasa Indonesia santai."},
                            {"role": "user", "content":
                                f"ISLM bergerak {change_pct:+.1f}% dari Rp {self.last_alert_price:,.0f} ke Rp {price:,.0f}. "
                                f"RSI={self._cache.get('rsi', 50):.0f}, Signal={sig.get('label', 'N/A')}, "
                                f"Trend={sig.get('trend', 'N/A')}, Whale={self._cache.get('whale_label', 'N/A')}"}
                        ],
                        max_tokens=150, temperature=0.7,
                    )
                    lines.append(f"\n🧠 *AI:* {resp.choices[0].message.content.strip()}")
            except: pass

            lines.append(f"\n⏰ {self._cache.get('last_update', '')} WIB")
            self.send_message("\n".join(lines))

            self.last_alert_price = price
            self.last_alert_time = now
            self.alerts_this_hour += 1
            safe_print(f"[ALERT] Price moved {change_pct:+.1f}% → Alert sent ({self.alerts_this_hour}/{self.max_alerts_per_hour})")

    # ============================================
    # V7: CUSTOM TARGET CHECKER
    # ============================================
    def _check_custom_targets(self):
        """Check if any custom price targets have been hit."""
        price = self._cache.get('price', 0)
        if not price or not self.price_targets:
            return

        for tgt in self.price_targets:
            if tgt['notified']:
                continue
            hit = False
            if tgt['direction'] == 'above' and price >= tgt['price']:
                hit = True
            elif tgt['direction'] == 'below' and price <= tgt['price']:
                hit = True

            if hit:
                tgt['notified'] = True
                arrow = "⬆️" if tgt['direction'] == 'above' else "⬇️"
                self.send_message(
                    f"🎯 *TARGET TERCAPAI!*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"{arrow} ISLM {'naik ke' if tgt['direction'] == 'above' else 'turun ke'} "
                    f"Rp {price:,.0f}\n"
                    f"🎯 Target: Rp {tgt['price']:,.0f}\n\n"
                    f"💡 Set target baru: /target [harga]"
                )
                self._stats['alerts_sent'] += 1
                safe_print(f"[TARGET] Hit! Rp {tgt['price']:,.0f}")

        # Clean up notified targets
        self.price_targets = [t for t in self.price_targets if not t['notified']]

    # ============================================
    # V6: VOLATILITY ADAPTIVE SYSTEM
    # ============================================
    def check_volatility(self):
        """Monitor volatility and adjust notification frequency."""
        price = self._cache.get('price', 0)
        if not price:
            return

        self.price_samples_5min.append(price)
        # Keep only last 30 samples (~15 min at 30s poll / ~5 min at 10s)
        if len(self.price_samples_5min) > 30:
            self.price_samples_5min = self.price_samples_5min[-30:]

        if len(self.price_samples_5min) < 5:
            return

        # Calculate short-term volatility
        prices = self.price_samples_5min
        avg = sum(prices) / len(prices)
        variance = sum((p - avg) ** 2 for p in prices) / len(prices)
        volatility_pct = (variance ** 0.5 / (avg + 1e-10)) * 100

        # Price range in recent samples
        price_range = ((max(prices) - min(prices)) / (min(prices) + 1e-10)) * 100

        was_volatile = self.volatility_mode

        # High volatility: variance > 0.5% OR range > 2%
        if volatility_pct > 0.5 or price_range > 2.0:
            if not self.volatility_mode:
                self.volatility_mode = True
                safe_print(f"[VOLATILITY] HIGH detected! Vol={volatility_pct:.2f}%, Range={price_range:.1f}%")
                self.send_message(
                    f"⚡ *VOLATILITY ALERT!*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📊 Market sedang *BERGEJOLAK!*\n"
                    f"📈 Volatilitas: {volatility_pct:.2f}%\n"
                    f"📏 Range: {price_range:.1f}%\n\n"
                    f"🤖 *Bot switch ke MODE AKTIF*\n"
                    f"  Update setiap 10 menit\n"
                    f"  Alert otomatis jika ±3%\n\n"
                    f"💡 _Tetap tenang, ikuti sinyal AI._"
                )
        elif volatility_pct < 0.2 and price_range < 1.0:
            if self.volatility_mode:
                self.volatility_mode = False
                safe_print(f"[VOLATILITY] Returned to NORMAL. Vol={volatility_pct:.2f}%")
                self.send_message(
                    f"✅ *Market Stabil*\n"
                    f"Volatilitas kembali normal ({volatility_pct:.2f}%).\n"
                    f"Bot kembali ke jadwal standar (30m/1h/daily)."
                )

    def send_volatile_update(self):
        """Quick update during high volatility periods."""
        c = self._cache
        sig = c['ai_signal']
        prices = self.price_samples_5min
        if not prices:
            return

        range_pct = ((max(prices) - min(prices)) / (min(prices) + 1e-10)) * 100
        recent_change = ((prices[-1] - prices[0]) / (prices[0] + 1e-10)) * 100 if len(prices) >= 2 else 0

        lines = [
            f"⚡ *VOLATILE MODE UPDATE*",
            f"━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 Harga: Rp {c['price']:,.0f}",
            f"📊 Range 5m: {range_pct:.1f}% | Arah: {recent_change:+.1f}%",
            f"📢 Signal: {sig['label']} ({sig['confidence']*100:.0f}%)",
            f"📈 Trend: {sig['trend']}",
            f"🐋 {c['whale_label']}",
        ]

        # Quick AI take
        try:
            api_key = Config.GROQ_API_KEY or os.environ.get('GROQ_API_KEY', '')
            if api_key:
                from groq import Groq
                client = Groq(api_key=api_key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content":
                            "Market ISLM sedang volatile. Beri 1 kalimat singkat: "
                            "aksi yang disarankan sekarang. Bahasa Indonesia."},
                        {"role": "user", "content":
                            f"ISLM Rp {c['price']:,.0f}, Range 5m={range_pct:.1f}%, "
                            f"Signal={sig['label']}, RSI={c['rsi']:.0f}"}
                    ],
                    max_tokens=80, temperature=0.7,
                )
                lines.append(f"\n🧠 {resp.choices[0].message.content.strip()}")
        except: pass

        lines.append(f"\n⏰ {c.get('last_update', '')} WIB | ⚡ Mode Aktif")
        self.send_message("\n".join(lines))

    # ============================================
    # V7: GITHUB REPO SECURITY MONITOR
    # ============================================
    def _check_github_repo(self):
        """Monitor GitHub repo for suspicious activity."""
        now = time.time()
        if now - self._last_repo_check < self._repo_check_interval:
            return
        self._last_repo_check = now

        try:
            headers = {'Accept': 'application/vnd.github.v3+json'}
            if self._github_token:
                headers['Authorization'] = f'token {self._github_token}'

            resp = requests.get(
                f'https://api.github.com/repos/{self._github_repo}',
                headers=headers, timeout=10
            )
            if resp.status_code != 200:
                return

            data = resp.json()
            watchers = data.get('watchers_count', 0)
            forks = data.get('forks_count', 0)
            stars = data.get('stargazers_count', 0)

            # First check — just store baseline
            if self._last_known_watchers == -1:
                self._last_known_watchers = watchers
                self._last_known_forks = forks
                self._last_known_stars = stars
                safe_print(f"[GITHUB] Baseline: {stars} stars, {forks} forks, {watchers} watchers")
                return

            alerts = []
            if watchers > self._last_known_watchers:
                diff = watchers - self._last_known_watchers
                alerts.append(f"👁️ Watchers: {self._last_known_watchers} → {watchers} (+{diff})")
            if forks > self._last_known_forks:
                diff = forks - self._last_known_forks
                alerts.append(f"🔱 Forks: {self._last_known_forks} → {forks} (+{diff})")
            if stars > self._last_known_stars:
                diff = stars - self._last_known_stars
                alerts.append(f"⭐ Stars: {self._last_known_stars} → {stars} (+{diff})")

            if alerts:
                self.send_message(
                    f"🚨 *GITHUB SECURITY ALERT*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"📁 Repo: {self._github_repo}\n\n"
                    f"{'\n'.join(alerts)}\n\n"
                    f"⚠️ Ada orang yang melihat/fork repo kamu!\n"
                    f"🛡️ Keamanan otomatis ditingkatkan.\n"
                    f"\n💡 _Cek: github.com/{self._github_repo}_"
                )
                self._stats['alerts_sent'] += 1
                safe_print(f"[GITHUB] ALERT! Changes detected: {', '.join(alerts)}")

            # Update baseline
            self._last_known_watchers = watchers
            self._last_known_forks = forks
            self._last_known_stars = stars

        except Exception as e:
            safe_print(f"[GITHUB-ERR] {e}")

    # ============================================
    # MAIN LOOP
    # ============================================
    def run(self):
        safe_print("=" * 55)
        safe_print("🤖 ISLM Monitor V7 — AI Genius Bot")
        safe_print(f"📡 Token: ...{self.token[-6:]}" if self.token else "❌ NO TOKEN!")
        safe_print(f"💬 Chat ID: {self.chat_id}")
        safe_print(f"⏱️ Intervals: 30min / 1h / Daily + Smart Alerts")
        safe_print(f"🧠 AI V7: Memory + Vision + Chain-of-Thought")
        safe_print(f"⚡ Volatility Mode: Adaptive (10min during turbulence)")
        safe_print(f"📱 Phone Lookup: Active")
        safe_print("=" * 55)

        start_health_server()

        if not self.token or not self.chat_id:
            safe_print("[FATAL] Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID!")
            return

        # Initial analysis
        safe_print("[INIT] Running first analysis...")
        if self.refresh_analysis():
            self.last_alert_price = self._cache.get('price', 0)
            self.send_message(
                "🟢 *ISLM Bot V7 AKTIF*\n\n"
                "🧠 *AI Engine V7:*\n"
                "  • AI Genius + Memory\n"
                "  • 🖼️ Analisa Gambar (kirim foto!)\n"
                "  • 📱 Cek Nomor HP\n"
                "  • Smart Price Alerts\n\n"
                "📡 *Notifikasi:*\n"
                "  • ⚡ 30m / 📊 1h / 📅 Daily\n"
                "  • 🚨 Alert ±3% / ⚡ Volatile Mode\n\n"
                "💬 Chat / kirim gambar / /cek [nomor]\n"
                "Ketik /menu untuk semua perintah."
            )
            self.send_1hour_update()
        else:
            self.send_message("⚠️ *Bot V7 Started* (menunggu data...)")

        now = time.time()
        self.last_30min = now
        self.last_1hour = now
        self.last_daily = now

        # Main loop
        while True:
            try:
                # --- Handle Telegram messages ---
                updates = self.get_updates()
                if updates.get('ok'):
                    for u in updates.get('result', []):
                        self.offset = u['update_id'] + 1
                        if 'callback_query' in u:
                            cb = u['callback_query']
                            try:
                                requests.post(f"{self.base_url}/answerCallbackQuery",
                                    json={'callback_query_id': cb['id']}, timeout=5)
                            except: pass
                            self.send_message(self.handle_text_question(f"/{cb.get('data', '')}"))
                        elif 'message' in u:
                            msg = u['message']
                            # V7: Handle photos for image analysis
                            if 'photo' in msg:
                                safe_print(f"[PHOTO] Received image")
                                caption = msg.get('caption', '')
                                result = self.handle_photo(msg['photo'], caption)
                                self.send_message(result)
                            # Handle text messages
                            elif msg.get('text'):
                                txt = msg['text']
                                safe_print(f"[MSG] {txt}")
                                self._stats['messages_handled'] += 1
                                reply = self.handle_text_question(txt)
                                # Send with inline keyboard for /menu
                                if t_lower := txt.lower().strip():
                                    if t_lower in ['/start', '/menu', '/help', 'help', 'menu']:
                                        self.send_message(reply)
                                        buttons = [
                                            [{"text": "📊 Status", "callback_data": "status"},
                                             {"text": "📎 Sinyal", "callback_data": "sinyal"},
                                             {"text": "🔮 Prediksi", "callback_data": "predict"}],
                                            [{"text": "💼 Portfolio", "callback_data": "portfolio"},
                                             {"text": "🎯 Targets", "callback_data": "target"},
                                             {"text": "📊 Stats", "callback_data": "stats"}],
                                            [{"text": "🐳 Whale", "callback_data": "whale"},
                                             {"text": "📈 Risk", "callback_data": "risk"},
                                             {"text": "📰 News", "callback_data": "news"}],
                                        ]
                                        self.send_message_with_keyboard("⬇️ *Pilih menu:*", buttons)
                                    else:
                                        self.send_message(reply)
                                else:
                                    self.send_message(reply)

                # --- Multi-interval notifications ---
                now = time.time()

                # V6: Smart Price Alert (every poll cycle)
                try:
                    if self._cache.get('price'):
                        self.check_price_alert()
                        self.check_volatility()
                        # V7: Check custom price targets
                        self._check_custom_targets()
                except Exception as e:
                    safe_print(f"[ALERT-ERR] {e}")
                    self._stats['errors'] += 1

                # V7: GitHub repo security check
                try:
                    self._check_github_repo()
                except Exception as e:
                    safe_print(f"[GITHUB-ERR] {e}")

                # Scheduled notifications (respect pause)
                if not self.notifications_paused:
                    # V6: Volatile mode — extra updates every 10 min
                    if self.volatility_mode and now - self.last_volatile_check >= self.volatile_interval:
                        safe_print("[VOLATILE] Sending volatile mode update...")
                        if self.refresh_analysis():
                            self.send_volatile_update()
                        self.last_volatile_check = now

                    # 30-minute update
                    if now - self.last_30min >= INTERVAL_30MIN:
                        safe_print("[30MIN] Sending quick update...")
                        if self.refresh_analysis():
                            self.send_30min_update()
                        self.last_30min = now

                    # 1-hour update
                    if now - self.last_1hour >= INTERVAL_1HOUR:
                        safe_print("[1HOUR] Sending full analysis...")
                        if self.refresh_analysis():
                            self.send_1hour_update()
                        self.last_1hour = now

                    # Daily update
                    if now - self.last_daily >= INTERVAL_DAILY:
                        safe_print("[DAILY] Sending daily recap...")
                        if self.refresh_analysis():
                            self.send_daily_update()
                        self.last_daily = now

            except KeyboardInterrupt:
                safe_print("\n[EXIT] Bot stopped.")
                self.send_message("🔴 *Bot V7 Stopped*")
                break
            except Exception as e:
                safe_print(f"[ERROR] {e}")
                traceback.print_exc()
                time.sleep(10)


if __name__ == "__main__":
    bot = StandaloneBot()
    bot.run()

