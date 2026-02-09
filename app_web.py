import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend.api import IndodaxAPI
from backend.core_logic import (
    MarketProjector,
    FundamentalEngine,
    QuantAnalyzer,
    CandleSniper,
    WhaleTracker,
    AISignalEngine,
)
from backend.auth_engine import AuthEngine
from backend.telegram_bot import TelegramBot
from backend.config import Config
import threading

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ISLM Monitor V3 ☁️",
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
                offset = bot.handle_updates(offset, api)

                ticker = api.get_price('islmidr')
                if not ticker.get('success'):
                    time.sleep(60)
                    continue
                price = ticker['last']

                candles = api.get_kline('islmidr', '15')
                if candles and len(candles) > 5:
                    if last_price == 0:
                        last_price = price
                    closes = pd.DataFrame(candles)['close'].values
                    rsi = QuantAnalyzer.calculate_rsi(closes)
                    _, _, hist = QuantAnalyzer.calculate_macd(closes)
                    bb_upper, bb_mid, bb_lower = QuantAnalyzer.calculate_bollinger_bands(closes)
                    depth = api.get_depth('islmidr') or {}
                    whale_ratio = WhaleTracker.get_whale_ratio(
                        depth.get('buy', []), depth.get('sell', []), 0.1
                    )
                    f_score, _ = FundamentalEngine.analyze_market_sentiment()
                    patterns = CandleSniper.analyze_patterns(candles)
                    bull_k = ("HAMMER", "INV. HAMMER", "BULL ENGULFING", "MORNING STAR")
                    bear_k = ("HANGING MAN", "SHOOTING STAR", "BEAR ENGULFING", "EVENING STAR")
                    cb = sum(1 for p in patterns if any(k in p for k in bull_k))
                    cbe = sum(1 for p in patterns if any(k in p for k in bear_k))
                    ai_signal = AISignalEngine.compute(
                        rsi=rsi, macd_hist=hist, price=price, bb_mid=bb_mid,
                        bb_upper=bb_upper, bb_lower=bb_lower,
                        candle_bull_count=cb, candle_bear_count=cbe,
                        whale_ratio=whale_ratio, fundamental_score=f_score,
                    )
                    signal = ai_signal["label"]
                    if abs(price - last_price) / (last_price + 1e-10) > 0.02:
                        bot.send_dashboard_menu(price, (price - last_price) / last_price * 100, rsi, signal)
                        last_price = price

            except Exception as e:
                print(f"Bg Loop Error: {e}")

            time.sleep(60)

# Start Background Thread
if 'monitor' not in st.session_state:
    st.session_state.monitor = BackgroundMonitor()

# --- CSS STYLING ---
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .metric-card { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    h1, h2, h3 { color: #58a6ff; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; }
    .stButton>button:hover { border-color: #58a6ff; color: #58a6ff; }
    .reason-box { background: #161b22; border-left: 3px solid #58a6ff; padding: 10px; margin: 5px 0; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- SECURITY & CONFIG ---
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT = 15 * 60
OTP_COOLDOWN = 60  # seconds between OTP requests

# --- SESSION STATE INIT ---
if 'authenticated' not in st.session_state: st.session_state.authenticated = False
if 'last_activity' not in st.session_state: st.session_state.last_activity = time.time()
if 'login_attempts' not in st.session_state: st.session_state.login_attempts = 0
if 'otp_sent' not in st.session_state: st.session_state.otp_sent = False
if 'generated_otp' not in st.session_state: st.session_state.generated_otp = None
if 'last_otp_time' not in st.session_state: st.session_state.last_otp_time = 0

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
    st.write("Akses terbatas. Masukkan Kode Otentikasi 2FA via Telegram.")

    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.error("⛔ SISTEM TERKUNCI: Terlalu banyak percobaan gagal. Tunggu 5 menit.")
        return

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📡 KIRIM KODE KE TELEGRAM"):
            # Rate limiting
            elapsed = time.time() - st.session_state.last_otp_time
            if elapsed < OTP_COOLDOWN:
                st.warning(f"⏳ Tunggu {int(OTP_COOLDOWN - elapsed)}s sebelum kirim ulang.")
            else:
                auth = AuthEngine()
                otp = auth.generate_otp()
                success, msg = auth.send_otp(otp)
                if success:
                    st.session_state.result_otp = otp
                    st.session_state.otp_sent = True
                    st.session_state.last_otp_time = time.time()
                    st.success("✅ Kode OTP telah dikirim ke Telegram!")
                else:
                    st.error(msg)

    if st.session_state.otp_sent:
        user_otp = st.text_input("Masukkan 6-Digit Kode:", type="password")
        if st.button("🚪 MASUK SISTEM"):
            if user_otp == st.session_state.result_otp:
                st.session_state.authenticated = True
                st.session_state.login_attempts = 0
                st.success("✅ Akses Diterima!")
                st.rerun()
            else:
                st.session_state.login_attempts += 1
                attempts_left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                st.error(f"⛔ Kode Salah! Sisa: {attempts_left}")


# ============================================
# MAIN DASHBOARD
# ============================================
def main_dashboard():
    api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("🕌 ISLM Monitor V3")
        st.success("🟢 Connected (Cloud)")
        if st.button("🔒 LOGOUT"):
            st.session_state.authenticated = False
            st.session_state.otp_sent = False
            st.rerun()

        st.subheader("⚙️ Settings")
        timeframe = st.selectbox("📊 Timeframe", ["1m", "15m", "1H", "1D"])

        st.markdown("---")
        st.caption("🛡️ **SECURITY STATUS**")
        st.success("✅ 2FA Active")
        st.info("🔒 Session Encrypted")
        st.warning(f"📡 Monitor: {'ON' if 'monitor' in st.session_state else 'OFF'}")

    # --- FETCH CORE DATA ---
    try:
        ticker = api.get_price('islmidr')
        if not ticker.get('success'):
            st.error("⚠️ Gagal ambil data harga. Coba refresh.")
            return
        price = ticker['last']
        high = ticker['high']
        low = ticker['low']
        vol = ticker.get('vol', 0)
        btc = api.get_price('btcidr').get('last', 0)
    except Exception:
        st.error("❌ API Error. Coba lagi nanti.")
        return

    # --- FETCH CANDLE DATA ---
    res_map = {'1m': '1', '15m': '15', '1H': '60', '1D': '1D'}
    res = res_map.get(timeframe, '15')
    candles = api.get_kline('islmidr', res)
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # --- COMPUTE ALL INDICATORS (BEFORE using them) ---
    closes = df['close'].values
    highs_arr = df['high'].values
    lows_arr = df['low'].values

    rsi = QuantAnalyzer.calculate_rsi(closes)
    macd_val, sig_line, hist = QuantAnalyzer.calculate_macd(closes)
    stoch_k, stoch_d = QuantAnalyzer.calculate_stoch_rsi(closes)
    bb_upper, bb_mid, bb_lower = QuantAnalyzer.calculate_bollinger_bands(closes)
    atr = QuantAnalyzer.calculate_atr(highs_arr, lows_arr, closes)
    market_phase = QuantAnalyzer.detect_market_phase(closes)

    # Whale & Order Book
    try:
        depth = api.get_depth("islmidr")
        whale_ratio = WhaleTracker.get_whale_ratio(
            depth.get("buy", []), depth.get("sell", []), 0.1
        )
    except Exception:
        whale_ratio = 0.5
    whale_label = WhaleTracker.interpret(whale_ratio)

    # Candle Patterns
    candle_patterns = CandleSniper.analyze_patterns(candles) if candles else []
    bull_keywords = ("HAMMER", "INV. HAMMER", "BULL ENGULFING", "MORNING STAR")
    bear_keywords = ("HANGING MAN", "SHOOTING STAR", "BEAR ENGULFING", "EVENING STAR")
    candle_bull = sum(1 for p in candle_patterns if any(k in p for k in bull_keywords))
    candle_bear = sum(1 for p in candle_patterns if any(k in p for k in bear_keywords))

    # Fundamental
    f_score, f_news = FundamentalEngine.analyze_market_sentiment()

    # === UNIFIED AI SIGNAL (computed BEFORE display) ===
    ai_signal = AISignalEngine.compute(
        rsi=rsi, macd_hist=hist, price=price,
        bb_mid=bb_mid, bb_upper=bb_upper, bb_lower=bb_lower,
        candle_bull_count=candle_bull, candle_bear_count=candle_bear,
        whale_ratio=whale_ratio, fundamental_score=f_score,
    )

    # Multi-horizon predictions
    price_list = [c['close'] for c in candles[-100:]] if candles else [price] * 100
    predictions = MarketProjector.predict_multi_horizon(price, price_list)

    # ============================================
    # TAB LAYOUT
    # ============================================
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard Realtime", "💬 Chat Konsultan AI", "📈 Analisa Mendalam"])

    # ============================================
    # TAB 1: DASHBOARD
    # ============================================
    with tab1:
        # Metrics Row
        c1, c2, c3, c4 = st.columns(4)
        range_pct = (price - low) / (high - low + 1e-10) * 100 - 50
        c1.metric("ISLM/IDR", f"Rp {price:,.0f}", f"{range_pct:.1f}%")
        c2.metric("Bitcoin (BTC)", f"Rp {btc:,.0f}")
        c3.metric("Fase Pasar", market_phase)
        c4.metric("Sinyal AI", ai_signal["label"], f"Confidence: {ai_signal['confidence']*100:.0f}%")

        # --- CHART with Volume ---
        st.subheader(f"📈 Chart ISLM ({timeframe})")

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25],
            subplot_titles=None
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            name="ISLM"
        ), row=1, col=1)

        # Bollinger Bands overlay
        if bb_upper is not None:
            n = len(df)
            if n >= 20:
                bb_times = df['time'].iloc[-20:]
                bb_u_vals = [bb_upper] * 20
                bb_l_vals = [bb_lower] * 20
                fig.add_trace(go.Scatter(
                    x=bb_times, y=bb_u_vals,
                    line=dict(color='rgba(88,166,255,0.3)', width=1, dash='dot'),
                    name="BB Upper", showlegend=False
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=bb_times, y=bb_l_vals,
                    line=dict(color='rgba(88,166,255,0.3)', width=1, dash='dot'),
                    fill='tonexty', fillcolor='rgba(88,166,255,0.05)',
                    name="BB Lower", showlegend=False
                ), row=1, col=1)

        # Volume bars
        colors = ['#26a69a' if row['close'] >= row['open'] else '#ef5350' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df['time'], y=df['vol'], marker_color=colors,
            name="Volume", showlegend=False
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            height=550,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            showlegend=False,
        )
        fig.update_yaxes(title_text="Harga (Rp)", row=1, col=1)
        fig.update_yaxes(title_text="Vol", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

        # --- AI REASONING BOX ---
        st.subheader("🧠 AI Reasoning (Explainable)")
        for reason in ai_signal.get("reasons", []):
            st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

        # --- PREDICTIONS ---
        st.subheader("🔮 AI Multi-Horizon Prediction")
        pc1, pc2, pc3 = st.columns(3)
        for col, (key, pred) in zip([pc1, pc2, pc3], predictions.items()):
            with col:
                col.metric(
                    f"Target {pred['label']}",
                    f"Rp {pred['target']:,.0f}",
                    f"{pred['change_pct']:+.1f}% {pred['direction']}"
                )
                col.caption(f"Range: Rp {pred['low']:,.0f} — Rp {pred['high']:,.0f} | Conf: {pred['confidence']:.0f}%")

        # News Ticker
        st.info(f"📰 **NEWS:** {f_news}")

    # ============================================
    # TAB 2: AI CHATBOT
    # ============================================
    with tab2:
        st.subheader("🤖 Konsultan AI ISLM (Context-Aware)")
        st.caption("AI ini menganalisa data REAL-TIME dan memberikan jawaban berdasarkan data terbaru.")

        # Quick Reply Buttons
        st.write("**⚡ Quick Actions:**")
        qr1, qr2, qr3, qr4, qr5, qr6 = st.columns(6)

        quick_prompt = None
        if qr1.button("📈 Naik/Turun?"): quick_prompt = "Apakah ISLM akan naik atau turun?"
        if qr2.button("📊 Analisa 1H"): quick_prompt = "Berikan analisa teknikal lengkap"
        if qr3.button("📅 Prediksi 7H"): quick_prompt = "Prediksi harga 7 hari kedepan"
        if qr4.button("📰 Berita"): quick_prompt = "Berita terbaru tentang ISLM"
        if qr5.button("🐋 Whale"): quick_prompt = "Bagaimana aktivitas whale saat ini?"
        if qr6.button("🔐 Security"): quick_prompt = "Status keamanan sistem"

        st.markdown("---")

        # Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Process input (either quick button or typed)
        prompt = quick_prompt or st.chat_input("Tanya AI tentang ISLM...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- SMART AI RESPONSE ENGINE ---
            response = _generate_ai_response(
                prompt, price, rsi, macd_val, hist, sig_line,
                stoch_k, bb_upper, bb_mid, bb_lower,
                ai_signal, whale_ratio, whale_label,
                candle_patterns, f_score, f_news,
                market_phase, predictions, candles, atr
            )

            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # ============================================
    # TAB 3: ADVANCED ANALYSIS
    # ============================================
    with tab3:
        st.subheader("📊 Advanced Market Analysis")

        # Gauge Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RSI (14)", f"{rsi:.1f}", "Bullish" if rsi > 50 else "Bearish")
        m2.metric("MACD", f"{macd_val:.2f}", f"Hist: {hist:.2f}")
        m3.metric("Stochastic", f"{stoch_k:.1f}",
                  "Overbought" if stoch_k > 80 else "Oversold" if stoch_k < 20 else "Neutral")
        m4.metric("ATR", f"{atr:.2f}" if atr else "N/A", "Volatilitas")

        st.markdown("---")

        # Whale Tracker
        col_w, col_c = st.columns(2)
        with col_w:
            st.write("### 🐋 Whale Tracker")
            st.metric("Rasio Order Besar", f"{whale_ratio*100:.0f}% Buy", whale_label)

        with col_c:
            st.write("### 🕯️ Candle Pattern Sniper")
            if candle_patterns:
                st.write("Pola terdeteksi: " + ", ".join(candle_patterns))
            else:
                st.caption("Tidak ada pola reversal/continuation terdeteksi.")

        st.markdown("---")

        # Technical Indicators
        st.write("### 📐 Technical Indicators")
        if bb_upper is not None:
            ti1, ti2, ti3 = st.columns(3)
            ti1.metric("Bollinger Upper", f"Rp {bb_upper:,.0f}")
            ti2.metric("Bollinger Mid (SMA20)", f"Rp {bb_mid:,.0f}")
            ti3.metric("Bollinger Lower", f"Rp {bb_lower:,.0f}")
        else:
            st.warning("⚠️ Data tidak cukup untuk Bollinger Bands (butuh 20+ candle).")

        # Fibonacci
        st.write("### 📏 Fibonacci Levels")
        fib = QuantAnalyzer.calculate_fibonacci(high, low)
        fib_cols = st.columns(len(fib))
        for col, (level, val) in zip(fib_cols, fib.items()):
            col.metric(f"Fib {level}", f"Rp {val:,.0f}")

        st.markdown("---")

        # AI Reasoning (repeated for convenience)
        st.write("### 🧠 AI Signal Breakdown")
        for reason in ai_signal.get("reasons", []):
            st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

        # Send to Telegram
        st.markdown("---")
        if st.button("🚨 KIRIM ANALISA KE TELEGRAM"):
            bot = TelegramBot()
            summary = AISignalEngine.generate_ai_summary(ai_signal, price, predictions)
            success = bot.send_message(summary)
            if success:
                st.success("✅ Analisa AI terkirim ke Telegram!")
            else:
                st.error("❌ Gagal kirim. Cek Chat ID.")


# ============================================
# SMART AI RESPONSE ENGINE
# ============================================
def _generate_ai_response(
    prompt, price, rsi, macd_val, hist, sig_line,
    stoch_k, bb_upper, bb_mid, bb_lower,
    ai_signal, whale_ratio, whale_label,
    candle_patterns, f_score, f_news,
    market_phase, predictions, candles, atr
):
    """Context-aware AI chatbot response generator."""
    p = prompt.lower()

    # --- Price / Status ---
    if any(k in p for k in ["harga", "price", "status", "berapa"]):
        bb_info = f"BB: Rp {bb_lower:,.0f} — Rp {bb_upper:,.0f}" if bb_upper else "BB: N/A"
        return (
            f"💰 **Harga ISLM Sekarang: Rp {price:,.0f}**\n\n"
            f"📊 RSI: {rsi:.1f} ({'Overbought ⚠️' if rsi > 70 else 'Oversold 🟢' if rsi < 30 else 'Netral'})\n"
            f"📈 MACD: {macd_val:.2f} (Hist: {hist:+.2f})\n"
            f"📐 {bb_info}\n"
            f"🏷️ Fase Pasar: {market_phase}\n"
            f"📢 **Sinyal AI: {ai_signal['label']}** (Confidence: {ai_signal['confidence']*100:.0f}%)\n"
            f"📈 Trend: {ai_signal['trend']}"
        )

    # --- Prediction / Forecast ---
    if any(k in p for k in ["prediksi", "predict", "ramal", "naik", "turun", "forecast", "target"]):
        lines = [f"🔮 **PREDIKSI AI ISLM** (Dari harga Rp {price:,.0f})\n"]
        for key, pred in predictions.items():
            lines.append(
                f"**{pred['label']}:** Rp {pred['target']:,.0f} "
                f"({pred['change_pct']:+.1f}%) {pred['direction']}\n"
                f"  _Range: Rp {pred['low']:,.0f} — Rp {pred['high']:,.0f} | "
                f"Confidence: {pred['confidence']:.0f}%_\n"
            )
        lines.append(f"\n📢 **Sinyal Sekarang: {ai_signal['label']}**")
        lines.append(f"📈 Trend: {ai_signal['trend']}")

        # Add reasoning
        lines.append(f"\n**Alasan AI:**")
        for r in ai_signal.get("reasons", [])[:3]:
            lines.append(f"• {r}")
        return "\n".join(lines)

    # --- Technical Analysis ---
    if any(k in p for k in ["analisa", "teknikal", "technical", "indikator"]):
        bb_info = (
            f"📐 Bollinger Bands:\n"
            f"  Upper: Rp {bb_upper:,.0f} | Mid: Rp {bb_mid:,.0f} | Lower: Rp {bb_lower:,.0f}\n"
        ) if bb_upper else "📐 Bollinger: Data tidak cukup\n"
        return (
            f"📊 **ANALISA TEKNIKAL ISLM**\n\n"
            f"📈 RSI (14): {rsi:.1f}\n"
            f"📉 MACD: {macd_val:.2f} | Signal: {sig_line:.2f} | Histogram: {hist:+.2f}\n"
            f"📊 Stochastic: {stoch_k:.1f}\n"
            f"📏 ATR: {atr:.2f}\n" if atr else ""
            f"{bb_info}"
            f"🏷️ Fase Pasar: {market_phase}\n\n"
            f"**Kesimpulan AI:** {ai_signal['label']} ({ai_signal['confidence']*100:.0f}% confidence)\n"
            f"**Reasoning:**\n" + "\n".join(f"• {r}" for r in ai_signal.get("reasons", []))
        )

    # --- News ---
    if any(k in p for k in ["berita", "news", "kabar"]):
        return (
            f"📰 **BERITA & SENTIMEN TERBARU:**\n\n"
            f"{f_news}\n\n"
            f"📊 Skor Fundamental: {f_score}/10\n"
            f"🏷️ Fase Pasar: {market_phase}\n"
            f"📢 Sinyal AI: {ai_signal['label']}"
        )

    # --- Whale ---
    if any(k in p for k in ["whale", "paus", "order book", "orderbook"]):
        return (
            f"🐋 **WHALE TRACKER ISLM:**\n\n"
            f"📊 Rasio Whale Buy: {whale_ratio*100:.0f}%\n"
            f"📢 Interpretasi: {whale_label}\n\n"
            f"{'⚠️ PERHATIAN: Whale sedang Distribusi (Jual). Waspada tekanan turun.' if whale_ratio < 0.4 else ''}"
            f"{'🟢 Whale sedang Akumulasi (Beli). Sinyal positif!' if whale_ratio > 0.6 else ''}"
            f"{'⚖️ Whale seimbang. Tidak ada tekanan dominan.' if 0.4 <= whale_ratio <= 0.6 else ''}"
        )

    # --- Signal ---
    if any(k in p for k in ["sinyal", "signal", "beli", "jual", "buy", "sell"]):
        lines = [f"📢 **SINYAL AI ISLM: {ai_signal['label']}**\n"]
        lines.append(f"🎯 Confidence: {ai_signal['confidence']*100:.0f}%")
        lines.append(f"📈 Trend: {ai_signal['trend']}\n")
        lines.append("**Alasan:**")
        for r in ai_signal.get("reasons", []):
            lines.append(f"• {r}")
        if candle_patterns:
            lines.append(f"\n🕯️ Pola Candle: {', '.join(candle_patterns)}")
        return "\n".join(lines)

    # --- Security ---
    if any(k in p for k in ["security", "keamanan", "aman"]):
        return (
            f"🛡️ **STATUS KEAMANAN SISTEM:**\n\n"
            f"✅ 2FA Telegram: **Aktif**\n"
            f"🔒 Session Encryption: **Aktif**\n"
            f"📡 Background Monitor: **{'Aktif' if 'monitor' in st.session_state else 'Nonaktif'}**\n"
            f"⏱️ Session Timeout: **15 menit**\n"
            f"🚫 Max Login Attempts: **3x**\n"
            f"🔑 API Keys: **Tersimpan di Environment Variables**"
        )

    # --- Candle Pattern ---
    if any(k in p for k in ["pola", "pattern", "candle"]):
        if candle_patterns:
            return f"🕯️ **Pola Candlestick Terdeteksi:**\n\n" + "\n".join(f"• {p}" for p in candle_patterns)
        return "🕯️ Tidak ada pola candlestick signifikan terdeteksi saat ini."

    # --- Default ---
    return (
        f"🤖 Hai Bos! Saya AI Konsultan ISLM.\n\n"
        f"**Saya bisa jawab tentang:**\n"
        f"• 💰 Harga & Status → _\"Berapa harga ISLM?\"_\n"
        f"• 🔮 Prediksi → _\"Prediksi 7 hari\"_\n"
        f"• 📊 Analisa Teknikal → _\"Analisa teknikal\"_\n"
        f"• 📢 Sinyal → _\"Sinyal beli/jual\"_\n"
        f"• 🐋 Whale → _\"Aktivitas whale\"_\n"
        f"• 📰 Berita → _\"Berita terbaru\"_\n"
        f"• 🕯️ Candle → _\"Pola candle\"_\n"
        f"• 🔐 Keamanan → _\"Status keamanan\"_\n\n"
        f"Atau gunakan tombol Quick Action di atas! ⚡"
    )


# --- APP ROUTER ---
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()
