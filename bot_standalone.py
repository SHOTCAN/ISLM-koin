"""
ISLM Monitor — Standalone Telegram AI Bot
==========================================
Script ini berjalan INDEPENDEN dari Streamlit.
Bisa dijalankan di VPS, PC, atau cloud server 24/7.

Fitur:
  - Auto-update setiap 5 menit ke Telegram
  - Handle pertanyaan manual via Telegram chat
  - Smart intent recognition (8 kategori)
  - Auto-recovery jika error

Cara jalankan:
  python bot_standalone.py
  ATAU
  run_bot.bat (Windows)
"""

import sys
import os
import time
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from backend.api import IndodaxAPI
from backend.config import Config
from backend.core_logic import (
    MarketProjector,
    FundamentalEngine,
    QuantAnalyzer,
    CandleSniper,
    WhaleTracker,
    AISignalEngine,
)

import requests
import json
from datetime import datetime

# ============================================
# CONFIG
# ============================================
UPDATE_INTERVAL = 300  # 5 minutes in seconds
POLL_INTERVAL = 3      # Check Telegram every 3 seconds
PAIR = 'islmidr'

# ============================================
# TELEGRAM BOT (Standalone Version)
# ============================================
class StandaloneBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = None
        self.last_update_time = 0
        self.api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)

        # Cache for latest analysis
        self._cache = {
            "price": 0,
            "rsi": 50,
            "macd_val": 0,
            "hist": 0,
            "sig_line": 0,
            "bb_upper": None,
            "bb_mid": None,
            "bb_lower": None,
            "whale_ratio": 0.5,
            "whale_label": "⚖️ Whale Seimbang",
            "ai_signal": {"label": "HOLD 🤝", "confidence": 0, "trend": "Sideways", "reasons": [], "score": 0},
            "predictions": {},
            "candle_patterns": [],
            "f_score": 0,
            "f_news": "",
            "market_phase": "NETRAL",
            "atr": None,
            "stoch_k": 50,
            "last_update": "Never",
        }

    def send_message(self, text, parse_mode="Markdown"):
        if not self.chat_id:
            print("[ERROR] Chat ID not set!")
            return False
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
            r = requests.post(url, json=payload, timeout=10)
            return r.status_code == 200
        except Exception as e:
            print(f"[SEND ERROR] {e}")
            return False

    def get_updates(self):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'timeout': POLL_INTERVAL, 'offset': self.offset}
            r = requests.get(url, params=params, timeout=POLL_INTERVAL + 5)
            return r.json()
        except:
            return {}

    # ============================================
    # CORE: Fetch & Analyze Market Data
    # ============================================
    def refresh_analysis(self):
        """Fetch latest market data and run full AI analysis."""
        try:
            # Price
            ticker = self.api.get_price(PAIR)
            if not ticker.get('success'):
                print("[WARN] Price fetch failed")
                return False
            price = ticker['last']
            high = ticker['high']
            low = ticker['low']

            # Candles
            candles = self.api.get_kline(PAIR, '15')
            if not candles or len(candles) < 5:
                print("[WARN] Candle data insufficient")
                return False

            df = pd.DataFrame(candles)
            closes = df['close'].values
            highs_arr = df['high'].values
            lows_arr = df['low'].values

            # Indicators
            rsi = QuantAnalyzer.calculate_rsi(closes)
            macd_val, sig_line, hist = QuantAnalyzer.calculate_macd(closes)
            stoch_k, _ = QuantAnalyzer.calculate_stoch_rsi(closes)
            bb_upper, bb_mid, bb_lower = QuantAnalyzer.calculate_bollinger_bands(closes)
            atr = QuantAnalyzer.calculate_atr(highs_arr, lows_arr, closes)
            market_phase = QuantAnalyzer.detect_market_phase(closes)

            # Whale
            try:
                depth = self.api.get_depth(PAIR) or {}
                whale_ratio = WhaleTracker.get_whale_ratio(
                    depth.get('buy', []), depth.get('sell', []), 0.1
                )
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

            # AI Signal
            ai_signal = AISignalEngine.compute(
                rsi=rsi, macd_hist=hist, price=price,
                bb_mid=bb_mid, bb_upper=bb_upper, bb_lower=bb_lower,
                candle_bull_count=cb, candle_bear_count=cbe,
                whale_ratio=whale_ratio, fundamental_score=f_score,
            )

            # Predictions
            price_list = [c['close'] for c in candles[-100:]]
            predictions = MarketProjector.predict_multi_horizon(price, price_list)

            # Update cache
            self._cache.update({
                "price": price,
                "rsi": rsi,
                "macd_val": macd_val,
                "hist": hist,
                "sig_line": sig_line,
                "bb_upper": bb_upper,
                "bb_mid": bb_mid,
                "bb_lower": bb_lower,
                "whale_ratio": whale_ratio,
                "whale_label": whale_label,
                "ai_signal": ai_signal,
                "predictions": predictions,
                "candle_patterns": candle_patterns,
                "f_score": f_score,
                "f_news": f_news,
                "market_phase": market_phase,
                "atr": atr,
                "stoch_k": stoch_k,
                "last_update": datetime.now().strftime('%H:%M:%S'),
            })

            print(f"[OK] Analysis updated | Price: Rp {price:,.0f} | Signal: {ai_signal['label']}")
            return True

        except Exception as e:
            print(f"[ERROR] Analysis failed: {e}")
            traceback.print_exc()
            return False

    # ============================================
    # AUTO-UPDATE: Send periodic analysis to Telegram
    # ============================================
    def send_auto_update(self):
        """Send full AI analysis to Telegram (called every 5 minutes)."""
        c = self._cache
        sig = c["ai_signal"]

        lines = []
        lines.append("🤖 *ISLM MONITOR — AUTO UPDATE*")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"💰 *Harga:* Rp {c['price']:,.0f}")
        lines.append(f"📢 *Sinyal:* {sig['label']} ({sig['confidence']*100:.0f}%)")
        lines.append(f"📈 *Trend:* {sig['trend']}")
        lines.append(f"🏷️ *Fase:* {c['market_phase']}")
        lines.append("")

        # Indicators
        lines.append("📊 *INDIKATOR:*")
        lines.append(f"  RSI: {c['rsi']:.1f} | MACD: {c['macd_val']:.2f}")
        lines.append(f"  Stoch: {c['stoch_k']:.1f} | ATR: {c['atr']:.2f}" if c['atr'] else f"  Stoch: {c['stoch_k']:.1f}")
        if c['bb_upper']:
            lines.append(f"  BB: {c['bb_lower']:,.0f} — {c['bb_upper']:,.0f}")
        lines.append("")

        # Predictions
        lines.append("🔮 *PREDIKSI:*")
        for key, pred in c["predictions"].items():
            lines.append(
                f"  {pred['label']}: Rp {pred['target']:,.0f} "
                f"({pred['change_pct']:+.1f}%) {pred['direction']}"
            )
        lines.append("")

        # Whale
        lines.append(f"🐋 *Whale:* {c['whale_label']} ({c['whale_ratio']*100:.0f}%)")

        # Candle Patterns
        if c['candle_patterns']:
            lines.append(f"🕯️ *Pola:* {', '.join(c['candle_patterns'])}")

        # Reasoning (top 3)
        lines.append("")
        lines.append("🧠 *REASONING:*")
        for r in sig.get("reasons", [])[:4]:
            lines.append(f"  • {r}")

        lines.append("")
        lines.append(f"⏰ {c['last_update']} WIB | Next: 5 min")

        self.send_message("\n".join(lines))

    # ============================================
    # MANUAL QUESTIONS: Smart Intent Recognition
    # ============================================
    def handle_text_question(self, text):
        """Process a text question and return AI response."""
        t = text.lower().strip()
        c = self._cache
        sig = c["ai_signal"]

        # --- /start or /menu ---
        if t in ['/start', '/menu', '/help', 'help', 'bantuan']:
            return (
                "🤖 *ISLM AI BOT — MENU*\n\n"
                "*Ketik atau klik:*\n"
                "📈 /status — Kondisi market\n"
                "🔮 /predict — Prediksi harga\n"
                "📊 /analisa — Analisa teknikal\n"
                "📰 /news — Berita & fundamental\n"
                "🐋 /whale — Aktivitas whale\n"
                "📢 /sinyal — Sinyal AI\n"
                "🔐 /security — Status keamanan\n"
                "🕯️ /candle — Pola candlestick\n\n"
                "💬 Atau tanya bebas: _\"ISLM naik gak?\"_"
            )

        # --- STATUS ---
        if any(k in t for k in ['/status', 'status', 'harga', 'price', 'berapa', 'market', 'kondisi']):
            bb_info = f"BB: Rp {c['bb_lower']:,.0f} — Rp {c['bb_upper']:,.0f}" if c['bb_upper'] else "BB: N/A"
            return (
                f"📊 *STATUS ISLM*\n\n"
                f"💰 Harga: Rp {c['price']:,.0f}\n"
                f"📢 Sinyal: {sig['label']} ({sig['confidence']*100:.0f}%)\n"
                f"📈 Trend: {sig['trend']}\n"
                f"🏷️ Fase: {c['market_phase']}\n\n"
                f"📊 RSI: {c['rsi']:.1f} | MACD: {c['macd_val']:.2f}\n"
                f"📐 {bb_info}\n"
                f"🐋 Whale: {c['whale_label']}\n\n"
                f"⏰ Update: {c['last_update']} WIB"
            )

        # --- PREDICTION ---
        if any(k in t for k in ['/predict', 'prediksi', 'ramal', 'forecast', 'target', 'naik', 'turun']):
            lines = [f"🔮 *PREDIKSI AI ISLM*\n💰 Harga: Rp {c['price']:,.0f}\n"]
            for key, pred in c["predictions"].items():
                lines.append(
                    f"*{pred['label']}:* Rp {pred['target']:,.0f} "
                    f"({pred['change_pct']:+.1f}%) {pred['direction']}\n"
                    f"  _Range: Rp {pred['low']:,.0f} — Rp {pred['high']:,.0f}_\n"
                    f"  _Confidence: {pred['confidence']:.0f}%_"
                )
            lines.append(f"\n📢 Sinyal: {sig['label']}")
            lines.append(f"\n🧠 Alasan:")
            for r in sig.get("reasons", [])[:3]:
                lines.append(f"• {r}")
            return "\n".join(lines)

        # --- TECHNICAL ANALYSIS ---
        if any(k in t for k in ['/analisa', 'analisa', 'teknikal', 'technical', 'indikator']):
            bb_info = (
                f"📐 Bollinger Bands:\n"
                f"  Upper: Rp {c['bb_upper']:,.0f}\n"
                f"  Mid: Rp {c['bb_mid']:,.0f}\n"
                f"  Lower: Rp {c['bb_lower']:,.0f}\n"
            ) if c['bb_upper'] else "📐 BB: Data kurang\n"
            atr_info = f"📏 ATR: {c['atr']:.2f}\n" if c['atr'] else ""
            return (
                f"📊 *ANALISA TEKNIKAL ISLM*\n\n"
                f"📈 RSI (14): {c['rsi']:.1f}\n"
                f"📉 MACD: {c['macd_val']:.2f} | Signal: {c['sig_line']:.2f} | Hist: {c['hist']:+.2f}\n"
                f"📊 Stochastic: {c['stoch_k']:.1f}\n"
                f"{atr_info}"
                f"{bb_info}\n"
                f"🏷️ Fase: {c['market_phase']}\n"
                f"📢 Sinyal: {sig['label']} ({sig['confidence']*100:.0f}%)"
            )

        # --- NEWS / FUNDAMENTAL ---
        if any(k in t for k in ['/news', 'berita', 'news', 'fundamental', 'kabar', 'sentimen']):
            return (
                f"📰 *BERITA & FUNDAMENTAL ISLM*\n\n"
                f"{c['f_news']}\n\n"
                f"📊 Skor Fundamental: {c['f_score']}/10\n"
                f"🏷️ Fase: {c['market_phase']}\n"
                f"📢 Sinyal: {sig['label']}"
            )

        # --- WHALE ---
        if any(k in t for k in ['/whale', 'whale', 'paus', 'order book']):
            detail = ""
            if c['whale_ratio'] > 0.6:
                detail = "⚡ Whale sedang AKUMULASI. Sinyal positif!"
            elif c['whale_ratio'] < 0.4:
                detail = "⚠️ Whale sedang DISTRIBUSI. Waspada tekanan turun."
            else:
                detail = "⚖️ Tidak ada tekanan dominan dari whale."
            return (
                f"🐋 *WHALE TRACKER ISLM*\n\n"
                f"📊 Rasio Whale Buy: {c['whale_ratio']*100:.0f}%\n"
                f"📢 {c['whale_label']}\n\n"
                f"{detail}"
            )

        # --- SIGNAL ---
        if any(k in t for k in ['/sinyal', 'sinyal', 'signal', 'beli', 'jual', 'buy', 'sell']):
            lines = [f"📢 *SINYAL AI: {sig['label']}*\n"]
            lines.append(f"🎯 Confidence: {sig['confidence']*100:.0f}%")
            lines.append(f"📈 Trend: {sig['trend']}\n")
            lines.append("🧠 *Alasan:*")
            for r in sig.get("reasons", []):
                lines.append(f"• {r}")
            if c['candle_patterns']:
                lines.append(f"\n🕯️ Pola: {', '.join(c['candle_patterns'])}")
            return "\n".join(lines)

        # --- SECURITY ---
        if any(k in t for k in ['/security', 'keamanan', 'aman', 'security']):
            return (
                f"🛡️ *STATUS KEAMANAN*\n\n"
                f"✅ Bot Standalone: *Aktif*\n"
                f"✅ Telegram 2FA: *Aktif*\n"
                f"🔑 API Keys: *Env Variables*\n"
                f"📡 Auto-Update: *Setiap 5 menit*\n"
                f"⏱️ Uptime sejak: {datetime.now().strftime('%H:%M')}"
            )

        # --- CANDLE PATTERN ---
        if any(k in t for k in ['/candle', 'pola', 'pattern', 'candle']):
            if c['candle_patterns']:
                return f"🕯️ *POLA CANDLESTICK:*\n\n" + "\n".join(f"• {p}" for p in c['candle_patterns'])
            return "🕯️ Tidak ada pola candlestick terdeteksi saat ini."

        # --- DEFAULT ---
        return (
            "🤖 Maaf, saya belum paham pertanyaan itu.\n\n"
            "Coba ketik:\n"
            "/status — Kondisi market\n"
            "/predict — Prediksi harga\n"
            "/analisa — Analisa teknikal\n"
            "/news — Berita\n"
            "/whale — Whale tracker\n"
            "/sinyal — Sinyal AI\n"
            "/security — Keamanan\n"
            "/candle — Pola candle"
        )

    # ============================================
    # MAIN LOOP
    # ============================================
    def run(self):
        """Main loop: auto-update every 5min + handle manual questions."""
        print("=" * 50)
        print("🤖 ISLM Monitor — Standalone Bot STARTED")
        print(f"📡 Token: ...{self.token[-6:]}" if self.token else "❌ NO TOKEN!")
        print(f"💬 Chat ID: {self.chat_id}")
        print(f"⏱️ Auto-update: Every {UPDATE_INTERVAL}s")
        print(f"📥 Poll interval: Every {POLL_INTERVAL}s")
        print("=" * 50)

        if not self.token or not self.chat_id:
            print("[FATAL] Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in .env!")
            return

        # Initial analysis
        print("[INIT] Running first analysis...")
        if self.refresh_analysis():
            self.send_message(
                "🟢 *ISLM Bot Standalone AKTIF*\n\n"
                "Bot sekarang berjalan independen.\n"
                "Auto-update setiap 5 menit.\n\n"
                "Ketik /menu untuk lihat perintah."
            )
            self.send_auto_update()
        else:
            self.send_message("⚠️ *Bot Started* (data belum tersedia, menunggu...)")

        self.last_update_time = time.time()

        # Main loop
        while True:
            try:
                # --- Handle incoming Telegram messages ---
                updates = self.get_updates()
                if updates.get('ok'):
                    for u in updates.get('result', []):
                        self.offset = u['update_id'] + 1

                        # Callback queries (button clicks)
                        if 'callback_query' in u:
                            cb = u['callback_query']
                            try:
                                requests.post(
                                    f"{self.base_url}/answerCallbackQuery",
                                    json={'callback_query_id': cb['id']},
                                    timeout=5
                                )
                            except:
                                pass
                            data = cb.get('data', '')
                            response = self.handle_text_question(f"/{data}")
                            self.send_message(response)

                        # Text messages
                        elif 'message' in u:
                            text = u['message'].get('text', '')
                            if text:
                                print(f"[MSG] Received: {text}")
                                response = self.handle_text_question(text)
                                self.send_message(response)

                # --- Auto-update every 5 minutes ---
                elapsed = time.time() - self.last_update_time
                if elapsed >= UPDATE_INTERVAL:
                    print(f"\n[AUTO] {UPDATE_INTERVAL}s elapsed, refreshing analysis...")
                    if self.refresh_analysis():
                        self.send_auto_update()
                    else:
                        print("[WARN] Analysis failed, will retry next cycle")
                    self.last_update_time = time.time()

            except KeyboardInterrupt:
                print("\n[EXIT] Bot stopped by user.")
                self.send_message("🔴 *Bot Stopped* — Manual shutdown.")
                break
            except Exception as e:
                print(f"[ERROR] Main loop: {e}")
                traceback.print_exc()
                time.sleep(10)  # Wait before retry


# ============================================
# ENTRY POINT
# ============================================
if __name__ == "__main__":
    bot = StandaloneBot()
    bot.run()
