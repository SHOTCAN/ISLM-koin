import requests
import json
import time
from backend.config import Config

class TelegramBot:
    def __init__(self):
        self.token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, text, parse_mode="Markdown"):
        if not self.chat_id: return False
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode}
            requests.post(url, json=payload, timeout=10)
            return True
        except Exception as e:
            print(f"TeleBot send Error: {e}")
            return False

    def send_dashboard_menu(self, price, change, rsi, signal):
        """Kirim menu dashboard dengan tombol interaktif."""
        if not self.chat_id: return False

        icon = "🟢" if change >= 0 else "🔴"
        trend = "Bullish 🐂" if rsi > 50 else "Bearish 🐻"

        msg = (
            f"{icon} *STATUS ISLM LIVE*\n"
            f"💰 Harga: Rp {price:,.0f} ({change:+.2f}%)\n"
            f"📊 RSI: {rsi:.1f} ({trend})\n"
            f"📢 Sinyal AI: *{signal}*\n\n"
            f"👇 *PILIH MENU DI BAWAH:* 👇"
        )

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "📊 Cek Status", "callback_data": "status"},
                    {"text": "🔮 Prediksi AI", "callback_data": "predict"}
                ],
                [
                    {"text": "📰 Berita Terbaru", "callback_data": "news"},
                    {"text": "🚨 Set Alert", "callback_data": "alert"}
                ],
                [
                    {"text": "🖥️ Buka Web Dashboard", "url": "https://share.streamlit.io/shotcan/islm-koin/main/app_web.py"}
                ]
            ]
        }

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": msg,
                "parse_mode": "Markdown",
                "reply_markup": json.dumps(keyboard)
            }
            requests.post(url, json=payload, timeout=10)
            return True
        except Exception as e:
            print(f"TeleBot Menu Error: {e}")
            return False

    def get_updates(self, offset=None):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'timeout': 5, 'offset': offset}
            r = requests.get(url, params=params, timeout=10)
            return r.json()
        except:
            return {}

    def handle_updates(self, offset=None, api=None):
        """
        Polling handler for button clicks and commands.
        Supports live data via `api` parameter.
        """
        updates = self.get_updates(offset)
        if not updates.get('ok'): return offset

        for u in updates.get('result', []):
            offset = u['update_id'] + 1

            # Handle Button Click (CallbackQuery)
            if 'callback_query' in u:
                cb = u['callback_query']
                cb_id = cb['id']
                data = cb['data']

                # Acknowledge button press
                try:
                    requests.post(
                        f"{self.base_url}/answerCallbackQuery",
                        json={'callback_query_id': cb_id},
                        timeout=5
                    )
                except:
                    pass

                # Smart responses with live data
                if data == "status":
                    self._handle_status(api)
                elif data == "predict":
                    self._handle_predict(api)
                elif data == "news":
                    self._handle_news()
                elif data == "alert":
                    self.send_message("🚨 *ALERT AKTIF*\nNotifikasi otomatis untuk pergerakan > 2% sudah ON.")

            # Handle Commands (/start, /menu, /predict)
            elif 'message' in u:
                text = u['message'].get('text', '')
                if text in ['/start', '/menu']:
                    self._handle_status(api)
                elif text == '/predict':
                    self._handle_predict(api)

        return offset

    def _handle_status(self, api):
        """Send live status to Telegram."""
        try:
            if api:
                from backend.core_logic import QuantAnalyzer, AISignalEngine, FundamentalEngine, WhaleTracker, CandleSniper
                import pandas as pd

                ticker = api.get_price('islmidr')
                price = ticker.get('last', 0)
                candles = api.get_kline('islmidr', '15')

                if candles and len(candles) > 5:
                    closes = pd.DataFrame(candles)['close'].values
                    rsi = QuantAnalyzer.calculate_rsi(closes)
                    _, _, hist = QuantAnalyzer.calculate_macd(closes)
                    bb_u, bb_m, bb_l = QuantAnalyzer.calculate_bollinger_bands(closes)
                    depth = api.get_depth('islmidr') or {}
                    wr = WhaleTracker.get_whale_ratio(depth.get('buy', []), depth.get('sell', []), 0.1)
                    fs, _ = FundamentalEngine.analyze_market_sentiment()
                    patterns = CandleSniper.analyze_patterns(candles)
                    bull_k = ("HAMMER", "INV. HAMMER", "BULL ENGULFING", "MORNING STAR")
                    bear_k = ("HANGING MAN", "SHOOTING STAR", "BEAR ENGULFING", "EVENING STAR")
                    cb = sum(1 for p in patterns if any(k in p for k in bull_k))
                    cbe = sum(1 for p in patterns if any(k in p for k in bear_k))

                    sig = AISignalEngine.compute(
                        rsi=rsi, macd_hist=hist, price=price,
                        bb_mid=bb_m, bb_upper=bb_u, bb_lower=bb_l,
                        candle_bull_count=cb, candle_bear_count=cbe,
                        whale_ratio=wr, fundamental_score=fs,
                    )
                    summary = AISignalEngine.generate_ai_summary(sig, price)
                    self.send_message(summary)
                    return
            self.send_message("📊 *STATUS*\nData tidak tersedia saat ini.")
        except Exception as e:
            self.send_message(f"📊 *STATUS*\nError: {str(e)[:100]}")

    def _handle_predict(self, api):
        """Send prediction results to Telegram."""
        try:
            if api:
                from backend.core_logic import MarketProjector
                ticker = api.get_price('islmidr')
                price = ticker.get('last', 0)
                candles = api.get_kline('islmidr', '15')
                prices = [c['close'] for c in candles[-100:]] if candles else [price] * 100
                preds = MarketProjector.predict_multi_horizon(price, prices)

                lines = [f"🔮 *PREDIKSI AI ISLM*\n💰 Harga: Rp {price:,.0f}\n"]
                for key, pred in preds.items():
                    lines.append(
                        f"*{pred['label']}:* Rp {pred['target']:,.0f} "
                        f"({pred['change_pct']:+.1f}%) {pred['direction']}"
                    )
                self.send_message("\n".join(lines))
                return
            self.send_message("🔮 *PREDIKSI*\nData tidak tersedia.")
        except Exception as e:
            self.send_message(f"🔮 *PREDIKSI*\nError: {str(e)[:100]}")

    def _handle_news(self):
        """Send news to Telegram."""
        try:
            from backend.core_logic import FundamentalEngine
            score, news = FundamentalEngine.analyze_market_sentiment()
            self.send_message(
                f"📰 *BERITA ISLM*\n\n{news}\n\n📊 Skor Fundamental: {score}/10"
            )
        except Exception as e:
            self.send_message(f"📰 *BERITA*\nError: {str(e)[:100]}")
