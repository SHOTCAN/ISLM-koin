import os
from pathlib import Path

# Load .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / '.env')
except ImportError:
    pass

# Try to load Streamlit secrets (only works inside Streamlit runtime)
_secrets = {}
try:
    import streamlit as st
    # Access each key individually to avoid FileNotFoundError on __len__
    _has_secrets = True
except Exception:
    _has_secrets = False


def _get(key, default=''):
    """Get config: Streamlit secrets → env vars → default."""
    # Try Streamlit secrets first (only on cloud)
    if _has_secrets:
        try:
            val = st.secrets.get(key)
            if val is not None:
                return val
        except Exception:
            pass
    # Fallback to env vars
    return os.getenv(key, default)


class Config:
    # Indodax API (Private)
    API_KEY = _get('INDODAX_API_KEY')
    SECRET_KEY = _get('INDODAX_SECRET_KEY')

    # Telegram Bot
    TELEGRAM_TOKEN = _get('TELEGRAM_TOKEN')
    TELEGRAM_CHAT_ID = _get('TELEGRAM_CHAT_ID')
