import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backend.api import IndodaxAPI
from backend.core_logic import MarketProjector, FundamentalEngine # Reuse existing logic
from backend.auth_engine import AuthEngine
from backend.config import Config

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ISLM Monitor Cloud ☁️",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS STYLING (Hacker/Crypto Theme) ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .stButton>button:hover { border-color: #58a6ff; color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# --- SECURITY & CONFIG ---
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT = 15 * 60 # 15 Minutes
LOCKDOWN_FILE = "lockdown.flag"

# --- AUTH ENTRIES ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'last_activity' not in st.session_state: st.session_state.last_activity = time.time()
if 'login_attempts' not in st.session_state: st.session_state.login_attempts = 0
if 'otp_sent' not in st.session_state: st.session_state.otp_sent = False
if 'generated_otp' not in st.session_state: st.session_state.generated_otp = None

# Check Timeout
if st.session_state.authenticated:
    if time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
        st.session_state.authenticated = False
        st.session_state.otp_sent = False
        st.error("⚠️ Sesi habis. Silakan login ulang.")
    else:
        st.session_state.last_activity = time.time()

def login_page():
    st.title("🔐 Security Checkpoint")
    st.write("Akses terbatas. Masukkan Kode Otentikasi dua arah (2FA).")
    
    # Lockdown Check
    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("⛔ SISTEM TERKUNCI SEMENTARA: Terlalu banyak percobaan gagal.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📡 KIRIM KODE KE TELEGRAM"):
            auth = AuthEngine()
            otp = auth.generate_otp()
            success, msg = auth.send_otp(otp)
            if success:
                st.session_state.result_otp = otp
                st.session_state.otp_sent = True
                st.success("Kode OTP telah dikirim ke Telegram Bos!")
            else:
                st.error(msg)
    
    if st.session_state.otp_sent:
        user_otp = st.text_input("Masukkan 6-Digit Kode:", type="password")
        if st.button("🚪 MASUK SISTEM"):
            if user_otp == st.session_state.result_otp:
                st.session_state.authenticated = True
                st.session_state.login_attempts = 0 # Reset
                st.success("Akses Diterima. Mengalihkan...")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                attempts_left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.error(f"⛔ Kode Salah! Sisa percobaan: {attempts_left}")

# --- MAIN DASHBOARD ---
def main_dashboard():
    # API Reuse
    api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)
    
    # Sidebar
    with st.sidebar:
        st.header("ISLM Monitor")
        st.success("🟢 Connected (Cloud)")
        if st.button("🔒 LOGOUT"):
            st.session_state.authenticated = False
            st.session_state.otp_sent = False
            st.rerun()
            
        st.subheader("Tools")
        timeframe = st.selectbox("Timeframe", ["15m", "1h", "4h", "1d"])
        
    # Main Header
    try:
        ticker = api.get_price('islmidr')
        price = ticker['last']
        high = ticker['high']
        low = ticker['low']
        btc = api.get_price('btcidr').get('last', 0)
    except:
        st.error("Gagal ambil data API.")
        return

    # Fundamental Logic Reuse
    f_score, f_news = FundamentalEngine.analyze_market_sentiment()
    
    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Realtime", "🤖 AI Consultant", "📈 Advanced Analysis"])

    with tab1:
        # Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ISLM Order", f"Rp {price:,.0f}", f"{(price-low)/(high-low)*100-50:.1f}%")
        c2.metric("Bitcoin (BTC)", f"Rp {btc:,.0f}")
        c3.metric("Fundamental Score", f"{f_score}/10", f_news[:20]+"...")
        c4.metric("AI Confidence", "Strong Buy" if f_score > 3 else "Neutral")

        # Charting (Plotly)
        st.subheader("📈 Realtime Chart")
        candles = api.get_kline('islmidr', timeframe.replace('m','').replace('h','')) # Simple adjustment
        if candles:
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            fig = go.Figure(data=[go.Candlestick(x=df['time'],
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'])])
            fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Data Chart belum tersedia. Coba refresh atau ganti timeframe.")

        # Prediction Section
        st.subheader("🔮 AI Future Prediction")
        sc1, sc2 = st.columns(2)
        if sc1.button("RAMAL 1 HARI"):
            with st.spinner("Simulating..."):
                mp = MarketProjector()
                # Reuse Monte Carlo Logic
                prices = [c['close'] for c in candles[-100:]] if candles else [price]*100
                vol = MarketProjector.calculate_volatility(np.array(prices))
                drift = MarketProjector.calculate_drift(np.array(prices))
                paths = MarketProjector.run_monte_carlo(price, vol, drift, 1440, 500)
                target = np.percentile(paths[:,-1], 50)
                st.success(f"Target 1 Hari: Rp {target:,.0f}")
                
        if sc2.button("RAMAL 2 MINGGU (Ramadhan)"):
             with st.spinner("Analisa Musiman..."):
                prices = [c['close'] for c in candles[-100:]] if candles else [price]*100
                vol = MarketProjector.calculate_volatility(np.array(prices))
                drift = MarketProjector.calculate_drift(np.array(prices))
                paths = MarketProjector.run_monte_carlo(price, vol, drift*1.2, 20160, 500)
                target = np.percentile(paths[:,-1], 50)
                st.success(f"Target Ramadhan: Rp {target:,.0f}")

        # News Ticker
        st.info(f"📰 **NEWS FLASH:** {f_news}")

    with tab2:
        st.subheader("🤖 AI Consultant (Beta)")
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Tanya AI tentang ISLM (Contoh: 'Harga sekarang?', 'Prediksi?', 'Saran?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # AI Logic (Simple Rule-Based for now)
            response = ""
            p_lower = prompt.lower()
            
            if "harga" in p_lower:
                response = f"Harga ISLM saat ini adalah **Rp {price:,.0f}**. High hari ini: Rp {high:,.0f}, Low: Rp {low:,.0f}."
            elif "prediksi" in p_lower or "ramal" in p_lower:
                response = f"Berdasarkan analisis Monte Carlo, ada potensi pergerakan ke arah **{(price-low)/(high-low)*100-50:.1f}%** dari range harian. Sentimen pasar saat ini: **{'Positif' if f_score > 0 else 'Negatif'}** ({f_score}/10)."
            elif "news" in p_lower or "berita" in p_lower:
                response = f"Berita terbaru: {f_news}"
            elif "saran" in p_lower or "beli" in p_lower or "jual" in p_lower:
                action = "BUY" if f_score > 3 else "WAIT/SELL"
                response = f"Saran AI saat ini: **{action}**. Fundamental Score: {f_score}/10. Selalu gunakan uang dingin ya Bos!"
            else:
                response = "Maaf, saya hanya bot AI spesialis ISLM. Tanya saya soal harga, prediksi, atau berita terkini! 🤖"

            with st.chat_message("assistant"): st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab3:
        st.subheader("📊 Advanced Market Data")
        st.write("Coming Soon: Order Book Depth, Whale Alert System, and Multi-Timeframe Analysis.")
        st.json(ticker)


# --- APP ROUTER ---
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()
