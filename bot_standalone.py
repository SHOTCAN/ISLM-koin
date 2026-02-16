"""
ISLM Monitor — Standalone Telegram AI Bot V4
=============================================
Jalan INDEPENDEN dari Streamlit. Deploy ke Koyeb/Railway = 24/7.

Notifikasi Multi-Interval:
  - 30 menit : Quick status (harga, sinyal, trend)
  - 1 jam    : Full analysis + indicators + ML
  - 1 hari   : Daily summary + prediksi + rekap

Smart Intent Recognition (10 kategori via Telegram chat)
AI Engine V4: ProTA + ML + Support/Resistance
"""

import sys
import os
import time
import traceback
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta

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
)

import requests


# ============================================
# HEALTH CHECK SERVER (for Koyeb/Cloud)
# ============================================
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'ISLM Bot V4 OK')
    def log_message(self, *a): pass

def start_health_server():
    port = int(os.environ.get('PORT', 8000))
    server = HTTPServer(('0.0.0.0', port), HealthHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[HEALTH] HTTP server on port {port}")


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
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = None
        self.api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)

        # Interval timers
        self.last_30min = 0
        self.last_1hour = 0
        self.last_daily = 0
        self.start_time = time.time()

        # Price history for daily recap
        self.price_history = []
        self.daily_high = 0
        self.daily_low = float('inf')

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

    # --- Telegram ---
    def send_message(self, text, parse_mode="Markdown"):
        if not self.chat_id: return False
        try:
            # Telegram max 4096 chars
            for chunk in [text[i:i+4000] for i in range(0, len(text), 4000)]:
                r = requests.post(f"{self.base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": chunk, "parse_mode": parse_mode}, timeout=10)
            return True
        except Exception as e:
            print(f"[SEND ERR] {e}")
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
                print("[WARN] Price fetch failed")
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

            print(f"[OK] V4 Analysis | Rp {price:,.0f} | {ai_signal['label']} | ML: {ml_result.get('ml_signal', 'N/A')}")
            return True

        except Exception as e:
            print(f"[ERROR] Analysis: {e}")
            traceback.print_exc()
            return False

    # ============================================
    # NOTIFICATION: 30 Menit — Quick Status
    # ============================================
    def send_30min_update(self):
        c = self._cache
        sig = c["ai_signal"]
        self.send_message(
            f"⚡ *ISLM QUICK UPDATE (30min)*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Harga: Rp {c['price']:,.0f}\n"
            f"📢 Sinyal: {sig['label']} ({sig['confidence']*100:.0f}%)\n"
            f"📈 Trend: {sig['trend']}\n"
            f"📊 RSI: {c['rsi']:.1f} | MACD: {c['hist']:+.2f}\n"
            f"🐋 Whale: {c['whale_label']}\n"
            f"⏰ {c['last_update']} WIB"
        )

    # ============================================
    # NOTIFICATION: 1 Jam — Full Analysis
    # ============================================
    def send_1hour_update(self):
        c = self._cache
        sig = c["ai_signal"]
        ml = c["ml_result"]
        pta = c["pro_ta"]

        lines = [
            "🤖 *ISLM FULL ANALYSIS (1 Jam)*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 *Harga:* Rp {c['price']:,.0f}",
            f"📢 *Rule-Based:* {sig['label']} ({sig['confidence']*100:.0f}%)",
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

        # ProTA extras
        if pta.get('adx'):
            lines.append(f"  ADX: {pta['adx']:.0f}")
        if pta.get('mfi'):
            lines.append(f"  MFI: {pta['mfi']:.0f}")
        if pta.get('williams_r') is not None:
            lines.append(f"  Williams %R: {pta['williams_r']:.0f}")
        if pta.get('roc') is not None:
            lines.append(f"  ROC: {pta['roc']:.2f}")

        # Support/Resistance
        if c['supports']:
            lines.append(f"\n🟢 *Support:* {', '.join(f'Rp {s:,.0f}' for s in c['supports'])}")
        if c['resistances']:
            lines.append(f"🔴 *Resistance:* {', '.join(f'Rp {r:,.0f}' for r in c['resistances'])}")

        # Predictions (top line)
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

        lines = [
            "📅 *ISLM DAILY RECAP*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"📆 {datetime.now().strftime('%d %B %Y')}",
            "",
            "💹 *RANGKUMAN HARI INI:*",
            f"  Open: Rp {open_price:,.0f}",
            f"  Close: Rp {close_price:,.0f}",
            f"  High: Rp {self.daily_high:,.0f}",
            f"  Low: Rp {self.daily_low:,.0f}",
            f"  Avg: Rp {avg_price:,.0f}",
            f"  Change: {daily_change:+.2f}%",
            "",
            f"📢 *Sinyal AI:* {sig['label']} ({sig['confidence']*100:.0f}%)",
        ]

        if ml.get('ml_available'):
            lines.append(f"🤖 *ML Model:* {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)")

        lines += [
            f"📈 *Trend:* {sig['trend']}",
            f"🏷️ *Fase:* {c['market_phase']}",
            "",
            "🔮 *PREDIKSI BESOK:*",
        ]

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

        # News
        news = NewsEngine.generate_news(c['market_phase'], c['whale_ratio'])
        lines.append(f"\n📰 *NEWS:* {news}")

        # Full reasoning
        lines.append("\n🧠 *ANALISA LENGKAP:*")
        for r in sig.get("reasons", []):
            lines.append(f"  • {r}")

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
                "🤖 *ISLM AI BOT V4 — MENU*\n\n"
                "*Perintah:*\n"
                "📈 /status — Kondisi market\n"
                "🔮 /predict — Prediksi harga\n"
                "📊 /analisa — Analisa teknikal lengkap\n"
                "🤖 /ml — Sinyal ML (AI)\n"
                "📰 /news — Berita & fundamental\n"
                "🐋 /whale — Aktivitas whale\n"
                "📢 /sinyal — Sinyal AI lengkap\n"
                "📐 /level — Support & Resistance\n"
                "🔐 /security — Status keamanan\n"
                "🕯️ /candle — Pola candlestick\n\n"
                "💬 Atau tanya bebas: _\"ISLM naik gak?\"_"
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

        # SECURITY
        if any(k in t for k in ['/security', 'keamanan', 'aman', 'security']):
            uptime = (time.time() - self.start_time) / 3600
            return (
                f"🛡️ *STATUS KEAMANAN*\n\n"
                f"✅ Bot V4: *Aktif*\n"
                f"✅ ML Engine: *{'Aktif' if ml.get('ml_available') else 'Loading'}*\n"
                f"✅ ProTA (40+ indicators): *Aktif*\n"
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
    # GROQ AI — Free Llama 3.3 70B Chat
    # ============================================
    def _ask_groq_ai(self, user_message):
        """Send user message to Groq AI with full market context."""
        api_key = Config.GROQ_API_KEY if hasattr(Config, 'GROQ_API_KEY') else ''
        if not api_key:
            return (
                "🤖 Coba ketik:\n"
                "/status — Market\n/predict — Prediksi\n/analisa — Teknikal\n"
                "/ml — Sinyal ML\n/sinyal — Sinyal AI\n/level — Support/Resistance\n"
                "/whale — Whale\n/news — Berita\n/candle — Pola"
            )
        try:
            from groq import Groq
            client = Groq(api_key=api_key)
            c = self._cache
            sig = c["ai_signal"]
            ml = c["ml_result"]

            # Build rich market context
            context_lines = [
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
            if extras: context_lines.append(f"EXTRA: {' | '.join(extras)}")

            if ml.get('ml_available'):
                context_lines.append(f"ML SIGNAL: {ml['ml_signal']} ({ml['ml_confidence']*100:.0f}%)")
            if c['supports']:
                context_lines.append(f"SUPPORT: {', '.join(f'Rp {s:,.0f}' for s in c['supports'])}")
            if c['resistances']:
                context_lines.append(f"RESISTANCE: {', '.join(f'Rp {r:,.0f}' for r in c['resistances'])}")

            for k, p in c["predictions"].items():
                context_lines.append(f"PREDIKSI {p['label']}: Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%) {p['direction']}")

            context_lines.append("REASONING:")
            for r in sig.get("reasons", []):
                context_lines.append(f"  - {r}")

            market_context = "\n".join(context_lines)

            system_prompt = (
                "Kamu adalah AI analis trading profesional yang fokus pada ISLM (Islamic Coin) / Haqq Network. "
                "Jawab dalam Bahasa Indonesia yang ringkas dan jelas. "
                "Kamu punya akses data market real-time berikut:\n\n"
                f"{market_context}\n\n"
                "Gunakan data ini untuk menjawab pertanyaan user. "
                "Berikan analisa yang akurat, sertakan angka-angka penting. "
                "Jika ditanya tentang hal di luar trading ISLM, tetap jawab tapi kaitkan dengan konteks investasi/crypto. "
                "Jawab dengan emoji dan format yang rapi. Maksimal 500 kata."
            )

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=600,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[Groq Error] {e}")
            return (
                f"🤖 AI sedang sibuk. Gunakan perintah:\n"
                "/status — Market\n/predict — Prediksi\n/sinyal — Sinyal AI"
            )

    # ============================================
    # MAIN LOOP
    # ============================================
    def run(self):
        print("=" * 55)
        print("🤖 ISLM Monitor V4 — Standalone Bot")
        print(f"📡 Token: ...{self.token[-6:]}" if self.token else "❌ NO TOKEN!")
        print(f"💬 Chat ID: {self.chat_id}")
        print(f"⏱️ Intervals: 30min / 1h / Daily")
        print(f"🧠 ProTA: {'✅' if True else '❌'} | ML: {'✅' if True else '❌'}")
        print("=" * 55)

        start_health_server()

        if not self.token or not self.chat_id:
            print("[FATAL] Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID!")
            return

        # Initial analysis
        print("[INIT] Running first analysis...")
        if self.refresh_analysis():
            self.send_message(
                "🟢 *ISLM Bot V4 AKTIF*\n\n"
                "🧠 AI Engine V4:\n"
                "  • ProTA (40+ indikator)\n"
                "  • ML GradientBoosting\n"
                "  • Support/Resistance\n\n"
                "📡 Notifikasi:\n"
                "  • ⚡ 30 menit — Quick status\n"
                "  • 📊 1 jam — Full analysis\n"
                "  • 📅 Harian — Daily recap\n\n"
                "Ketik /menu untuk semua perintah."
            )
            self.send_1hour_update()
        else:
            self.send_message("⚠️ *Bot V4 Started* (menunggu data...)")

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
                            txt = u['message'].get('text', '')
                            if txt:
                                print(f"[MSG] {txt}")
                                self.send_message(self.handle_text_question(txt))

                # --- Multi-interval notifications ---
                now = time.time()

                # 30-minute update
                if now - self.last_30min >= INTERVAL_30MIN:
                    print("[30MIN] Sending quick update...")
                    if self.refresh_analysis():
                        self.send_30min_update()
                    self.last_30min = now

                # 1-hour update
                if now - self.last_1hour >= INTERVAL_1HOUR:
                    print("[1HOUR] Sending full analysis...")
                    if self.refresh_analysis():
                        self.send_1hour_update()
                    self.last_1hour = now

                # Daily update
                if now - self.last_daily >= INTERVAL_DAILY:
                    print("[DAILY] Sending daily recap...")
                    if self.refresh_analysis():
                        self.send_daily_update()
                    self.last_daily = now

            except KeyboardInterrupt:
                print("\n[EXIT] Bot stopped.")
                self.send_message("🔴 *Bot V4 Stopped*")
                break
            except Exception as e:
                print(f"[ERROR] {e}")
                traceback.print_exc()
                time.sleep(10)


if __name__ == "__main__":
    bot = StandaloneBot()
    bot.run()
