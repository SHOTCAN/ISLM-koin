import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / '.env')

class Config:
    API_KEY = os.getenv('INDODAX_API_KEY', '')
    SECRET_KEY = os.getenv('INDODAX_SECRET_KEY', '')
    PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'admin123')
    FLASK_SECRET = os.getenv('SECRET_KEY', 'dev-secret-key')
    HOST = os.getenv('HOST', '127.0.0.1')
    PORT = int(os.getenv('PORT', 5500))
    
    # TELEGRAM BOT
    TELEGRAM_TOKEN = "8519352553:AAE4pdqy7m8OC6BAXsaay2GSZ2SbVyPOBWM"
    TELEGRAM_CHAT_ID = None # Will be auto-discovered
