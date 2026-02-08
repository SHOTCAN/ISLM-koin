import random
import time
import requests
from backend.config import Config

class AuthEngine:
    """
    Mesin Keamanan Utama.
    Mengurus OTP via Telegram dan Session Validation.
    """
    def __init__(self):
        self.bot_token = Config.TELEGRAM_TOKEN
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.otp_storage = {} # Store OTP per session/IP temp

    def generate_otp(self):
        return str(random.randint(100000, 999999))

    def send_otp(self, otp_code):
        if not self.chat_id:
            return False, "Chat ID Telegram belum diset! Jalankan versi Desktop dulu untuk connect."
        
        msg = (
            f"🔐 *ISLM MONITOR WEB ACCESS*\n"
            f"Kode Rahasia: `{otp_code}`\n"
            f"Jangan berikan ke siapapun!"
        )
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": msg, "parse_mode": "Markdown"}
            requests.post(url, json=payload, timeout=5)
            return True, "Kode OTP terkirim ke Telegram!"
        except Exception as e:
            return False, f"Gagal kirim OTP: {e}"

    def verify_otp(self, input_otp, real_otp):
        return input_otp == real_otp
