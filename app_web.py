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

# --- AUTHENTICATION ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'otp_sent' not in st.session_state:
    st.session_state.otp_sent = False
if 'generated_otp' not in st.session_state:
    st.session_state.generated_otp = None

def login_page():
    st.title("🔐 Security Checkpoint")
    st.write("Akses terbatas. Masukkan Kode Otentikasi dua arah (2FA).")
    
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
                st.success("Akses Diterima. Mengalihkan...")
                st.rerun()
            else:
                st.error("⛔ Kode Salah! Akses Ditolak.")

# --- MAIN DASHBOARD ---
def main_dashboard():
    # API Reuse
    api = IndodaxAPI(Config.INDODAX_API_KEY, Config.INDODAX_SECRET_KEY)
    
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
    
    # Prediction Section
    st.subheader("🔮 AI Future Prediction")
    sc1, sc2 = st.columns(2)
    if sc1.button("RAMAL 1 HARI"):
        with st.spinner("Simulating..."):
            mp = MarketProjector()
            # Reuse Monte Carlo Logic (Simplified for Web)
            # Need to adapt MarketProjector to be static or instanceable easily without GUI
            prices = [c['close'] for c in candles[-100:]]
            vol = MarketProjector.calculate_volatility(np.array(prices))
            drift = MarketProjector.calculate_drift(np.array(prices))
            paths = MarketProjector.run_monte_carlo(price, vol, drift, 1440, 500)
            target = np.percentile(paths[:,-1], 50)
            st.success(f"Target 1 Hari: Rp {target:,.0f}")
            
    if sc2.button("RAMAL 2 MINGGU (Ramadhan)"):
         with st.spinner("Analisa Musiman..."):
            prices = [c['close'] for c in candles[-100:]]
            vol = MarketProjector.calculate_volatility(np.array(prices))
            drift = MarketProjector.calculate_drift(np.array(prices))
            # Boost drift for Ramadan
            paths = MarketProjector.run_monte_carlo(price, vol, drift*1.2, 20160, 500)
            target = np.percentile(paths[:,-1], 50)
            st.success(f"Target Ramadhan: Rp {target:,.0f}")

    # News Ticker
    st.info(f"📰 **NEWS FLASH:** {f_news}")


# --- APP ROUTER ---
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()
