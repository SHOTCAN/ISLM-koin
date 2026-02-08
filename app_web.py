import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from backend.api import IndodaxAPI
from backend.core_logic import MarketProjector, FundamentalEngine, QuantAnalyzer # Reuse existing logic
from backend.auth_engine import AuthEngine
from backend.telegram_bot import TelegramBot
from backend.config import Config
import threading

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ISLM Monitor Cloud ☁️",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- BACKGROUND MONITOR (24/7 BOT) ---
@st.cache_resource
class BackgroundMonitor:
    def __init__(self):
        self.running = True
        self.thread = threading.Thread(target=self.run_loop, daemon=True)
        self.thread.start()
    
    def run_loop(self):
        bot = TelegramBot()
        api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)
        last_price = 0
        offset = None 
        
        while self.running:
            try:
                # 0. Handle Telegram Updates (Buttons/Commands)
                offset = bot.handle_updates(offset)
                
                # 1. Fetch Real Data
                ticker = api.get_price('islmidr')
                price = ticker['last']
                
                # 2. Check RSI for Signal
                candles = api.get_kline('islmidr', '15')
                if candles:
                    closes = pd.DataFrame(candles)['close'].values
                    rsi = QuantAnalyzer.calculate_rsi(closes)
                    signal = "STRONG BUY 🚀" if rsi < 30 else "SELL ⚠️" if rsi > 70 else "HOLD 🤝"
                    
                    # 3. Send Periodic Update (Every 1 Hour or if Big Move)
                    # For demo/testing: We can make it more frequent or on-demand
                    # Here we simulate an "Alert" rule:
                    if abs(price - last_price) / last_price > 0.02: # 2% Move
                         bot.send_dashboard_menu(price, (price-last_price)/last_price*100, rsi, signal)
                         last_price = price
                         
            except Exception as e:
                print(f"Bg Loop Error: {e}")
            
            time.sleep(60) # Cek setiap 1 menit

# Start Background Thread
if 'monitor' not in st.session_state:
    st.session_state.monitor = BackgroundMonitor()

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
        
        # Determine timeframe resolution
        res_map = {'15m': '15', '1h': '60', '4h': '240', '1d': '1D'}
        res = res_map.get(timeframe, '15')
        
        candles = api.get_kline('islmidr', res)
        
        if candles:
            df = pd.DataFrame(candles)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # Basic Line Chart if Candle data is weak, else Candlestick
            fig = go.Figure(data=[go.Candlestick(x=df['time'],
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'],
                            increasing_line_color= '#26a69a', decreasing_line_color= '#ef5350')])
            
            fig.update_layout(
                template="plotly_dark", 
                height=500, 
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                margin=dict(l=0,r=0,t=0,b=0),
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- CALCULATE INDICATORS FOR AI & ADVANCED TAB ---
            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # RSI & MACD
            rsi = QuantAnalyzer.calculate_rsi(closes)
            macd, sig, hist = QuantAnalyzer.calculate_macd(closes)
            stoch_k, stoch_d = QuantAnalyzer.calculate_stoch_rsi(closes)
            bb_upper, bb_mid, bb_lower = QuantAnalyzer.calculate_bollinger_bands(closes)
            
        else:
            st.warning("⚠️ Data Chart belum tersedia. Indodax mungkin sedang sibuk. Coba refresh 1 menit lagi.")
            rsi, macd, sig, hist, stoch_k, stoch_d, bb_upper, bb_mid, bb_lower = 50, 0, 0, 0, 50, 50, price*1.05, price, price*0.95 # Default safe values

        # Prediction Section
        st.subheader("🔮 AI Future Prediction")
        sc1, sc2 = st.columns(2)
        if sc1.button("RAMAL 1 HARI"):
            with st.spinner("Simulating..."):
                mp = MarketProjector()
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
                # Boost drift for Ramadan
                paths = MarketProjector.run_monte_carlo(price, vol, drift*1.2, 20160, 500)
                target = np.percentile(paths[:,-1], 50)
                st.success(f"Target Ramadhan: Rp {target:,.0f}")

        # News Ticker
        st.info(f"📰 **NEWS FLASH:** {f_news}")

    with tab2:
        st.subheader("🤖 AI Consultant (Persistent)")
        # Load chat history from session state (Memory only for now to keep it fast)
        # Future: Save to JSON if needed, but SessionState is enough for "Active Session"
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat Input
        if prompt := st.chat_input("Tanya AI tentang ISLM..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)

            # AI Logic 
            response = ""
            p_lower = prompt.lower()
            
            if "harga" in p_lower:
                response = f"Harga ISLM saat ini **Rp {price:,.0f}**. RSI: {rsi:.1f} ({'Overbought' if rsi>70 else 'Oversold' if rsi<30 else 'Netral'})."
            elif "prediksi" in p_lower:
                # Ensure 'vol' is defined, if candles were not available, it won't be from the prediction section.
                # For this case, we'll re-calculate or use a default if needed.
                # Assuming 'vol' is calculated in tab1's prediction section, it should be available.
                # If candles were not available, 'vol' would not be calculated. Let's ensure it's handled.
                if 'vol' not in locals(): # Check if vol was calculated in the prediction section
                    prices_for_vol = [c['close'] for c in candles[-100:]] if candles else [price]*100
                    vol = MarketProjector.calculate_volatility(np.array(prices_for_vol))
                response = f"Simulasi AI memproyeksikan volatilitas mingguan sebesar {vol*100:.2f}%. Tren jangka pendek: {'Bullish' if price > bb_mid else 'Bearish'}."
            elif "news" in p_lower:
                response = f"Berita: {f_news}"
            elif "analisa" in p_lower:
                 response = f"**Analisa Teknikal:**\n- RSI: {rsi:.1f}\n- MACD: {macd:.2f}\n- Stoch: {stoch_k:.1f}\n- Posisi: {'Diatas' if price > bb_mid else 'Dibawah'} Bollinger Tengah."
            else:
                response = "Saya Siap! Tanya saya soal 'Harga', 'Prediksi', 'News', atau 'Analisa'."

            with st.chat_message("assistant"): st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    with tab3:
        st.subheader("📊 Advanced Market Analysis")
        if candles:
            # Gauge Charts or Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("RSI (14)", f"{rsi:.1f}", "Bullish" if rsi > 50 else "Bearish")
            m2.metric("MACD Level", f"{macd:.2f}", f"{hist:.2f}")
            m3.metric("Stochastic", f"{stoch_k:.1f}", "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral")
            
            st.markdown("---")
            st.write("### 📐 Technical Indicators")
            st.write(f"- **Bollinger Upper:** Rp {bb_upper:,.0f}")
            st.write(f"- **Bollinger Lower:** Rp {bb_lower:,.0f}")
            st.write(f"- **Signal Line:** {sig:.2f}")
            
            st.info("💡 **AI INSIGHT:** " + ("Pasar sedang Jenuh Beli (Hati-hati Koreksi)" if rsi > 70 else "Pasar Jenuh Jual (Potensi Rebound)" if rsi < 30 else "Pasar Sideways/Stabil."))
            
            if st.button("🚨 KIRIM SINYAL KE TELEGRAM"):
                bot = TelegramBot()
                signal = "STRONG BUY 🚀" if rsi < 30 else "SELL ⚠️" if rsi > 70 else "HOLD 🤝"
                change = (price - closes[-2])/closes[-2]*100 if len(closes) > 1 else 0
                success = bot.send_dashboard_menu(price, change, rsi, signal)
                if success:
                    st.success("Sinyal + Menu Terkirim ke Telegram!")
                else:
                    st.error("Gagal kirim. Cek Chat ID.")
        else:
            st.error("Data Candle belum tersedia untuk analisis mendalam.")


# --- APP ROUTER ---
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()
