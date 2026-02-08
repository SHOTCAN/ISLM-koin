import requests
import json
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
            requests.post(url, json=payload, timeout=5)
            return True
        except:
            return False

    def send_dashboard_menu(self, price, change, rsi, signal):
        """
        Kirim menu dashboard dengan tombol interaktif.
        """
        if not self.chat_id: return False
        
        # Emoji status
        icon = "🟢" if change >= 0 else "🔴"
        trend = "Bullish 🐂" if rsi > 50 else "Bearish 🐻"
        
        msg = (
            f"{icon} *STATUS ISLM LIVE*\n"
            f"💰 Harga: Rp {price:,.0f} ({change:+.2f}%)\n"
            f"📊 RSI: {rsi:.1f} ({trend})\n"
            f"📢 Sinyal AI: *{signal}*\n\n"
            f"👇 *PILIH MENU DI BAWAH:* 👇"
        )
        
        # Inline Keyboard (Tombol)
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
            requests.post(url, json=payload, timeout=5)
            return True
        except Exception as e:
            print(f"TeleBot Error: {e}")
            return False
