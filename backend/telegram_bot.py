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

    def get_updates(self, offset=None):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {'timeout': 100, 'offset': offset}
            r = requests.get(url, params=params, timeout=10)
            return r.json()
        except:
            return {}

    def handle_updates(self, offset=None):
        """
        Simple polling to handle button clicks.
        """
        updates = self.get_updates(offset)
        if not updates.get('ok'): return offset
        
        for u in updates.get('result', []):
            offset = u['update_id'] + 1
            
            # 1. Handle Button Click (CallbackQuery)
            if 'callback_query' in u:
                cb = u['callback_query']
                cb_id = cb['id']
                data = cb['data']
                chat_id = cb['message']['chat']['id']
                
                # Acknowledge (Stop loading animation)
                requests.post(f"{self.base_url}/answerCallbackQuery", json={'callback_query_id': cb_id})
                
                # Logic
                if data == "status":
                    self.send_message("📊 *LIVE STATUS ISLM*\nLoad data terakhir dari web...")
                elif data == "predict":
                    self.send_message("🔮 *AI PREDIKSI*\nAnalisa Monte Carlo sedang berjalan di server...")
                elif data == "news":
                    self.send_message("📰 *BERITA*\nCek Dashboard untuk berita lengkap.")
                elif data == "alert":
                    self.send_message("🚨 *ALERT SET*\nNotifikasi harga aktif untuk pergerakan >2%.")
            
            # 2. Handle Commands (/start, /menu)
            elif 'message' in u:
                msg = u['message']
                text = msg.get('text', '')
                if text == '/start' or text == '/menu':
                    self.send_dashboard_menu(0, 0, 50, "WAITING DATA...")
                    
        return offset
