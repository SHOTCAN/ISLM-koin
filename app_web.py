import streamlit as st
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from backend.api import IndodaxAPI
from backend.core_logic import (
    MarketProjector, FundamentalEngine, QuantAnalyzer,
    CandleSniper, WhaleTracker, AISignalEngine,
    ProTA, MLSignalClassifier, SupportResistance, NewsEngine,
)
from backend.auth_engine import AuthEngine
from backend.telegram_bot import TelegramBot
from backend.config import Config
import threading

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ISLM Monitor V4",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- PREMIUM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #0a0e17; color: #c9d1d9; }

    /* Header gradient */
    .header-bar {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 50%, #0d4b3c 100%);
        padding: 20px 30px; border-radius: 16px;
        border: 1px solid rgba(88,166,255,0.15);
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .header-bar h1 { color: #7ee8c7; margin: 0; font-size: 1.6rem; font-weight: 700; }
    .header-bar p { color: #8b949e; margin: 4px 0 0 0; font-size: 0.85rem; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #161b22, #1c2333);
        padding: 16px; border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.75rem; font-weight: 500; }
    [data-testid="stMetricValue"] { color: #e6edf3; font-weight: 600; }

    /* Signal badge */
    .signal-badge {
        display: inline-block; padding: 6px 16px; border-radius: 8px;
        font-weight: 600; font-size: 0.9rem;
    }
    .signal-buy { background: linear-gradient(135deg, #0a3d1c, #155d2e); color: #7ee8c7; border: 1px solid #238636; }
    .signal-sell { background: linear-gradient(135deg, #3d0a0a, #5d1515); color: #f85149; border: 1px solid #da3633; }
    .signal-hold { background: linear-gradient(135deg, #2d2a0a, #4d4515); color: #d29922; border: 1px solid #9e6a03; }

    /* Reason box */
    .reason-box {
        background: #161b22; border-left: 3px solid #238636;
        padding: 10px 14px; margin: 4px 0; border-radius: 6px;
        font-size: 0.85rem; color: #c9d1d9;
    }

    /* AI chat */
    .ai-context {
        background: linear-gradient(135deg, #0d1b2a, #0d2818);
        border: 1px solid rgba(126,232,199,0.2);
        padding: 14px; border-radius: 10px; margin: 8px 0;
        font-size: 0.85rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #161b22; border-radius: 8px; border: 1px solid #30363d;
        padding: 8px 16px; color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d4b3c, #1b5e4a); color: #7ee8c7;
        border-color: #238636;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 8px; font-weight: 500; border: 1px solid #30363d;
        background: #161b22; color: #c9d1d9; transition: all 0.2s;
    }
    .stButton>button:hover { border-color: #7ee8c7; color: #7ee8c7; background: #1c2333; }

    /* Hide Streamlit extras */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- SECURITY ---
MAX_LOGIN_ATTEMPTS = 3
SESSION_TIMEOUT = 15 * 60
OTP_COOLDOWN = 60
LOCKOUT_TIME = 300  # 5 min lockout

# --- SESSION STATE ---
defaults = {
    'authenticated': False, 'last_activity': time.time(),
    'login_attempts': 0, 'otp_sent': False,
    'generated_otp': None, 'last_otp_time': 0,
    'lockout_until': 0, 'messages': [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Session Timeout
if st.session_state.authenticated:
    if time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
        st.session_state.authenticated = False
        st.session_state.otp_sent = False
    else:
        st.session_state.last_activity = time.time()


# ============================================
# LOGIN PAGE (Enhanced Security)
# ============================================
def login_page():
    st.markdown("""
    <div style="display:flex;justify-content:center;align-items:center;min-height:60vh;">
    <div style="background:linear-gradient(145deg,#161b22,#1c2333);padding:40px;border-radius:16px;
    border:1px solid #30363d;max-width:420px;width:100%;box-shadow:0 8px 32px rgba(0,0,0,0.5);">
    <h2 style="color:#7ee8c7;text-align:center;margin:0 0 4px 0;">🔐 ISLM Monitor</h2>
    <p style="color:#8b949e;text-align:center;font-size:0.8rem;margin-bottom:24px;">
    Akses terbatas — Autentikasi 2FA via Telegram</p>
    </div></div>
    """, unsafe_allow_html=True)

    # Lockout check
    if time.time() < st.session_state.lockout_until:
        remaining = int(st.session_state.lockout_until - time.time())
        st.error(f"⛔ Terkunci. Tunggu {remaining} detik.")
        return

    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
        st.session_state.lockout_until = time.time() + LOCKOUT_TIME
        st.session_state.login_attempts = 0
        st.error("⛔ Terlalu banyak percobaan. Terkunci 5 menit.")
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📡 KIRIM KODE OTP KE TELEGRAM", use_container_width=True):
            elapsed = time.time() - st.session_state.last_otp_time
            if elapsed < OTP_COOLDOWN:
                st.warning(f"⏳ Tunggu {int(OTP_COOLDOWN - elapsed)}s")
            else:
                auth = AuthEngine()
                otp = auth.generate_otp()
                success, msg = auth.send_otp(otp)
                if success:
                    st.session_state.result_otp = otp
                    st.session_state.otp_sent = True
                    st.session_state.last_otp_time = time.time()
                    st.success("✅ Kode dikirim ke Telegram!")
                else:
                    st.error(msg)

        if st.session_state.otp_sent:
            user_otp = st.text_input("6-Digit Kode:", type="password", max_chars=6)
            if st.button("🚪 MASUK", use_container_width=True):
                if user_otp == st.session_state.result_otp:
                    st.session_state.authenticated = True
                    st.session_state.login_attempts = 0
                    st.rerun()
                else:
                    st.session_state.login_attempts += 1
                    left = MAX_LOGIN_ATTEMPTS - st.session_state.login_attempts
                    st.error(f"❌ Salah! Sisa: {left}")


# ============================================
# GROQ AI CHATBOT (100% FREE — Llama 3.3 70B)
# ============================================
def _ai_chat(prompt, market_context):
    """Send prompt to Groq API (free) with market context."""
    api_key = Config._get_config('GROQ_API_KEY', '')
    if not api_key:
        return None  # Fallback to rule-based

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        system_prompt = (
            "Kamu adalah AI analis trading profesional yang fokus pada ISLM (Islamic Coin) / Haqq Network. "
            "Jawab dalam Bahasa Indonesia yang ringkas dan jelas. "
            "Kamu punya akses data market real-time berikut:\n\n"
            f"{market_context}\n\n"
            "Gunakan data ini untuk menjawab pertanyaan user. "
            "Berikan analisa yang akurat, sertakan angka-angka penting. "
            "Jika ditanya tentang hal di luar trading ISLM, tetap jawab tapi kaitkan dengan konteks investasi/crypto. "
            "Format jawaban dengan emoji dan markdown yang rapi."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq Error] {e}")
        return None


def _build_market_context(price, rsi, macd_val, hist, stoch_k, bb_upper, bb_mid, bb_lower,
                          ai_signal, whale_ratio, whale_label, market_phase, predictions,
                          pro_ta, ml_result, supports, resistances, f_score, atr):
    """Build market context string for DeepSeek."""
    lines = [
        f"HARGA: Rp {price:,.0f}",
        f"SINYAL AI: {ai_signal['label']} (Confidence: {ai_signal['confidence']*100:.0f}%)",
        f"TREND: {ai_signal['trend']}",
        f"FASE: {market_phase}",
        f"RSI: {rsi:.1f} | MACD: {macd_val:.2f} | MACD Hist: {hist:+.2f} | Stoch: {stoch_k:.1f}",
    ]
    if bb_upper: lines.append(f"BB: Rp {bb_lower:,.0f} — Rp {bb_upper:,.0f}")
    if atr: lines.append(f"ATR: {atr:.2f}")
    lines.append(f"WHALE: {whale_label} ({whale_ratio*100:.0f}% buy)")
    lines.append(f"FUNDAMENTAL: {f_score}/10")

    if pro_ta:
        extras = []
        if pro_ta.get('adx'): extras.append(f"ADX={pro_ta['adx']:.0f}")
        if pro_ta.get('mfi'): extras.append(f"MFI={pro_ta['mfi']:.0f}")
        if pro_ta.get('williams_r') is not None: extras.append(f"Williams%R={pro_ta['williams_r']:.0f}")
        if pro_ta.get('ema_9') and pro_ta.get('ema_21'):
            extras.append(f"EMA9={'>' if pro_ta['ema_9'] > pro_ta['ema_21'] else '<'}EMA21")
        if extras: lines.append(f"EXTRA: {' | '.join(extras)}")

    if ml_result and ml_result.get('ml_available'):
        lines.append(f"ML SIGNAL: {ml_result['ml_signal']} ({ml_result['ml_confidence']*100:.0f}%)")

    if supports: lines.append(f"SUPPORT: {', '.join(f'Rp {s:,.0f}' for s in supports)}")
    if resistances: lines.append(f"RESISTANCE: {', '.join(f'Rp {r:,.0f}' for r in resistances)}")

    for k, p in predictions.items():
        lines.append(f"PREDIKSI {p['label']}: Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%) {p['direction']}")

    lines.append("REASONING AI:")
    for r in ai_signal.get("reasons", []):
        lines.append(f"  - {r}")

    return "\n".join(lines)


# ============================================
# MAIN DASHBOARD
# ============================================
def main_dashboard():
    api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)

    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown('<h3 style="color:#7ee8c7;">🕌 ISLM Monitor V4</h3>', unsafe_allow_html=True)
        st.success("🟢 Online")
        if st.button("🔒 Logout"):
            st.session_state.authenticated = False
            st.session_state.otp_sent = False
            st.session_state.messages = []
            st.rerun()
        timeframe = st.selectbox("📊 Timeframe", ["1m", "15m", "1H", "1D"], index=1)
        st.markdown("---")
        st.caption("🛡️ Security: 2FA Active")

    # --- FETCH DATA ---
    try:
        ticker = api.get_price('islmidr')
        if not ticker.get('success'):
            st.error("⚠️ API gagal. Refresh halaman.")
            return
        price = ticker['last']
        high, low = ticker['high'], ticker['low']
    except:
        st.error("❌ Gagal terhubung ke Indodax.")
        return

    res_map = {'1m': '1', '15m': '15', '1H': '60', '1D': '1D'}
    candles = api.get_kline('islmidr', res_map.get(timeframe, '15'))
    df = pd.DataFrame(candles)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    closes = df['close'].values
    highs_arr, lows_arr = df['high'].values, df['low'].values

    # --- INDICATORS (V4: ProTA + ML) ---
    rsi = QuantAnalyzer.calculate_rsi(closes)
    macd_val, sig_line, hist = QuantAnalyzer.calculate_macd(closes)
    stoch_k, _ = QuantAnalyzer.calculate_stoch_rsi(closes)
    bb_upper, bb_mid, bb_lower = QuantAnalyzer.calculate_bollinger_bands(closes)
    atr = QuantAnalyzer.calculate_atr(highs_arr, lows_arr, closes)
    market_phase = QuantAnalyzer.detect_market_phase(closes)

    pro_ta = ProTA.compute_all(df)
    ml_result = MLSignalClassifier.predict_signal(df)
    supports, resistances = SupportResistance.find_levels(df)

    # Override with ProTA
    if pro_ta.get('rsi'): rsi = pro_ta['rsi']
    if pro_ta.get('macd_hist'): hist = pro_ta['macd_hist']
    if pro_ta.get('bb_upper'): bb_upper = pro_ta['bb_upper']
    if pro_ta.get('bb_mid'): bb_mid = pro_ta['bb_mid']
    if pro_ta.get('bb_lower'): bb_lower = pro_ta['bb_lower']

    try:
        depth = api.get_depth("islmidr") or {}
        whale_ratio = WhaleTracker.get_whale_ratio(depth.get("buy", []), depth.get("sell", []), 0.1)
    except:
        whale_ratio = 0.5
    whale_label = WhaleTracker.interpret(whale_ratio)

    candle_patterns = CandleSniper.analyze_patterns(candles) if candles else []
    bull_k = ("HAMMER", "INV. HAMMER", "BULL ENGULFING", "MORNING STAR")
    bear_k = ("HANGING MAN", "SHOOTING STAR", "BEAR ENGULFING", "EVENING STAR")
    cb = sum(1 for p in candle_patterns if any(k in p for k in bull_k))
    cbe = sum(1 for p in candle_patterns if any(k in p for k in bear_k))
    f_score, f_news = FundamentalEngine.analyze_market_sentiment()

    ai_signal = AISignalEngine.compute(
        rsi=rsi, macd_hist=hist, price=price,
        bb_mid=bb_mid, bb_upper=bb_upper, bb_lower=bb_lower,
        candle_bull_count=cb, candle_bear_count=cbe,
        whale_ratio=whale_ratio, fundamental_score=f_score,
        pro_ta=pro_ta, ml_result=ml_result,
    )

    price_list = [c['close'] for c in candles[-100:]] if candles else [price] * 100
    predictions = MarketProjector.predict_multi_horizon(price, price_list)

    # --- HEADER ---
    sig = ai_signal
    sig_class = "signal-buy" if "BUY" in sig['label'] else "signal-sell" if "SELL" in sig['label'] else "signal-hold"
    st.markdown(f"""
    <div class="header-bar">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div>
                <h1>🕌 ISLM Monitor V4</h1>
                <p>AI-Powered • ProTA 40+ Indicators • ML GradientBoosting • Groq Llama AI (Free)</p>
            </div>
            <div style="text-align:right;">
                <div style="color:#e6edf3;font-size:1.8rem;font-weight:700;">Rp {price:,.0f}</div>
                <span class="signal-badge {sig_class}">{sig['label']}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "💬 AI Chat (Groq Free)", "📈 Analisa"])

    # ============================================
    # TAB 1: DASHBOARD
    # ============================================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        range_pct = (price - low) / (high - low + 1e-10) * 100 - 50
        c1.metric("ISLM/IDR", f"Rp {price:,.0f}", f"{range_pct:+.1f}%")
        c2.metric("Fase", market_phase)
        c3.metric("AI Sinyal", sig['label'], f"{sig['confidence']*100:.0f}%")
        if ml_result.get('ml_available'):
            c4.metric("ML Model", ml_result['ml_signal'], f"{ml_result['ml_confidence']*100:.0f}%")
        else:
            c4.metric("RSI", f"{rsi:.1f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Netral")

        # --- PREMIUM CHART ---
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.02, row_heights=[0.78, 0.22],
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df['time'], open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing=dict(line=dict(color='#00d4aa', width=1.2), fillcolor='#00d4aa'),
            decreasing=dict(line=dict(color='#ff4757', width=1.2), fillcolor='#ff4757'),
            name="ISLM", whiskerwidth=0.8,
        ), row=1, col=1)

        # EMA 9 & 21 overlay
        if pro_ta.get('ema_9') and len(df) >= 21:
            try:
                import ta as ta_lib
                ema9 = ta_lib.trend.ema_indicator(df['close'], window=9)
                ema21 = ta_lib.trend.ema_indicator(df['close'], window=21)
                fig.add_trace(go.Scatter(
                    x=df['time'], y=ema9,
                    line=dict(color='#ffa502', width=1.2), name="EMA 9", opacity=0.8,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df['time'], y=ema21,
                    line=dict(color='#3742fa', width=1.2), name="EMA 21", opacity=0.8,
                ), row=1, col=1)
            except:
                pass

        # Bollinger Bands
        if bb_upper is not None and len(df) >= 20:
            try:
                import ta as ta_lib
                bb = ta_lib.volatility.BollingerBands(df['close'], window=20)
                fig.add_trace(go.Scatter(
                    x=df['time'], y=bb.bollinger_hband(),
                    line=dict(color='rgba(126,232,199,0.25)', width=1, dash='dot'),
                    name="BB Upper", showlegend=False,
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df['time'], y=bb.bollinger_lband(),
                    line=dict(color='rgba(126,232,199,0.25)', width=1, dash='dot'),
                    fill='tonexty', fillcolor='rgba(126,232,199,0.04)',
                    name="BB Lower", showlegend=False,
                ), row=1, col=1)
            except:
                pass

        # Support/Resistance lines
        for s in supports:
            fig.add_hline(y=s, line_dash="dash", line_color="rgba(0,212,170,0.4)",
                          annotation_text=f"S: {s:,.0f}", row=1, col=1)
        for r in resistances:
            fig.add_hline(y=r, line_dash="dash", line_color="rgba(255,71,87,0.4)",
                          annotation_text=f"R: {r:,.0f}", row=1, col=1)

        # Volume
        colors = ['#00d4aa' if row['close'] >= row['open'] else '#ff4757' for _, row in df.iterrows()]
        fig.add_trace(go.Bar(
            x=df['time'], y=df['vol'], marker_color=colors,
            marker_opacity=0.6, name="Volume", showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark", height=520,
            plot_bgcolor="#0a0e17", paper_bgcolor="#0a0e17",
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="left", x=0,
                        font=dict(size=10, color="#8b949e"), bgcolor="rgba(0,0,0,0)"),
            xaxis2=dict(showgrid=False),
            yaxis=dict(gridcolor="rgba(48,54,61,0.4)", title=None),
            yaxis2=dict(gridcolor="rgba(48,54,61,0.2)", title=None),
        )
        st.plotly_chart(fig, use_container_width=True, key="main_chart")

        # --- PREDICTIONS ---
        st.markdown("#### 🔮 Prediksi AI")
        pc1, pc2, pc3 = st.columns(3)
        for col, (k, p) in zip([pc1, pc2, pc3], predictions.items()):
            col.metric(p['label'], f"Rp {p['target']:,.0f}", f"{p['change_pct']:+.1f}% {p['direction']}")
            col.caption(f"Range: Rp {p['low']:,.0f} — Rp {p['high']:,.0f}")

        # --- REASONING ---
        with st.expander("🧠 AI Reasoning", expanded=False):
            for r in sig.get("reasons", []):
                st.markdown(f'<div class="reason-box">{r}</div>', unsafe_allow_html=True)

    # ============================================
    # TAB 2: DEEPSEEK AI CHATBOT
    # ============================================
    with tab2:
        st.markdown("#### 💬 Tanya AI tentang ISLM")
        st.caption("Powered by Groq AI (Llama 3.3 70B) — 100% GRATIS, jawab kontekstual dengan data real-time")

        # Quick buttons
        qr1, qr2, qr3, qr4 = st.columns(4)
        quick = None
        if qr1.button("📈 Naik/Turun?"): quick = "Apakah ISLM akan naik atau turun? Berikan analisa lengkap."
        if qr2.button("🔮 Prediksi"): quick = "Berikan prediksi harga ISLM 1,3,7 hari dengan reasoning."
        if qr3.button("📊 Analisa"): quick = "Analisa teknikal ISLM secara mendalam."
        if qr4.button("🐋 Whale"): quick = "Bagaimana aktivitas whale ISLM saat ini?"

        st.markdown("---")

        # Chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = quick or st.chat_input("Tanya AI tentang ISLM trading...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Build market context
            context = _build_market_context(
                price, rsi, macd_val, hist, stoch_k, bb_upper, bb_mid, bb_lower,
                ai_signal, whale_ratio, whale_label, market_phase, predictions,
                pro_ta, ml_result, supports, resistances, f_score, atr
            )

            # Try DeepSeek first, fallback to rule-based
            with st.chat_message("assistant"):
                with st.spinner("🤖 AI sedang menganalisa..."):
                    response = _ai_chat(prompt, context)
                    if response is None:
                        response = _fallback_response(
                            prompt, price, rsi, macd_val, hist, stoch_k,
                            bb_upper, bb_mid, bb_lower, ai_signal, whale_ratio,
                            whale_label, market_phase, predictions, candle_patterns,
                            f_score, f_news, atr, pro_ta, ml_result, supports, resistances
                        )
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # ============================================
    # TAB 3: ADVANCED ANALYSIS
    # ============================================
    with tab3:
        st.markdown("#### 📊 Indikator Teknikal V4")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RSI", f"{rsi:.1f}", "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Netral")
        m2.metric("MACD", f"{macd_val:.2f}", f"Hist: {hist:+.2f}")
        m3.metric("Stochastic", f"{stoch_k:.1f}")
        m4.metric("ATR", f"{atr:.2f}" if atr else "N/A")

        # ProTA extras
        if pro_ta:
            st.markdown("---")
            st.markdown("#### 💪 ProTA (40+ Indicators)")
            p1, p2, p3, p4 = st.columns(4)
            if pro_ta.get('adx'):
                p1.metric("ADX", f"{pro_ta['adx']:.0f}", "Trend Kuat" if pro_ta['adx'] > 25 else "Sideways")
            if pro_ta.get('mfi'):
                p2.metric("MFI", f"{pro_ta['mfi']:.0f}")
            if pro_ta.get('williams_r') is not None:
                p3.metric("Williams %R", f"{pro_ta['williams_r']:.0f}")
            if pro_ta.get('roc') is not None:
                p4.metric("ROC", f"{pro_ta['roc']:.2f}")

            if pro_ta.get('ema_9') and pro_ta.get('ema_21'):
                cross = "✨ Golden Cross" if pro_ta['ema_9'] > pro_ta['ema_21'] else "💀 Death Cross"
                st.info(f"📊 EMA Cross: {cross} (EMA9={pro_ta['ema_9']:,.0f} vs EMA21={pro_ta['ema_21']:,.0f})")

        # ML Signal
        if ml_result.get('ml_available'):
            st.markdown("---")
            st.markdown("#### 🤖 ML Signal Classifier")
            st.metric("ML Prediction", ml_result['ml_signal'], f"{ml_result['ml_confidence']*100:.0f}% confidence")
            if ml_result.get('ml_class_probs'):
                for cls, prob in ml_result['ml_class_probs'].items():
                    bar = "█" * int(prob * 30) + "░" * (30 - int(prob * 30))
                    st.text(f"  {cls}: {bar} {prob*100:.0f}%")

        # Support / Resistance
        st.markdown("---")
        st.markdown("#### 📐 Support & Resistance")
        sr1, sr2 = st.columns(2)
        with sr1:
            if supports:
                for s in supports:
                    st.success(f"🟢 Support: Rp {s:,.0f}")
            else:
                st.caption("Belum cukup data")
        with sr2:
            if resistances:
                for r in resistances:
                    st.error(f"🔴 Resistance: Rp {r:,.0f}")
            else:
                st.caption("Belum cukup data")

        # Whale
        st.markdown("---")
        wc1, wc2 = st.columns(2)
        wc1.metric("🐋 Whale", f"{whale_ratio*100:.0f}% Buy", whale_label)
        if candle_patterns:
            wc2.info(f"🕯️ Pola: {', '.join(candle_patterns)}")
        else:
            wc2.caption("🕯️ Tidak ada pola terdeteksi")

        # Reasoning
        st.markdown("---")
        st.markdown("#### 🧠 AI Signal Breakdown")
        for r in sig.get("reasons", []):
            st.markdown(f'<div class="reason-box">{r}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("📡 Kirim Analisa ke Telegram", use_container_width=True):
            bot = TelegramBot()
            summary = AISignalEngine.generate_ai_summary(ai_signal, price, predictions)
            if bot.send_message(summary):
                st.success("✅ Terkirim!")
            else:
                st.error("❌ Gagal. Cek Chat ID.")


# ============================================
# FALLBACK RESPONSE (when DeepSeek unavailable)
# ============================================
def _fallback_response(prompt, price, rsi, macd_val, hist, stoch_k,
                       bb_upper, bb_mid, bb_lower, ai_signal, whale_ratio,
                       whale_label, market_phase, predictions, candle_patterns,
                       f_score, f_news, atr, pro_ta, ml_result, supports, resistances):
    p = prompt.lower()
    sig = ai_signal
    ml = ml_result

    if any(k in p for k in ["harga", "price", "status", "berapa"]):
        bb = f"BB: Rp {bb_lower:,.0f} — Rp {bb_upper:,.0f}" if bb_upper else "BB: N/A"
        r = f"💰 **ISLM: Rp {price:,.0f}**\n\n📢 {sig['label']} ({sig['confidence']*100:.0f}%)\n📈 {sig['trend']}\n📊 RSI: {rsi:.1f} | MACD: {hist:+.2f}\n📐 {bb}\n🐋 {whale_label}"
        if ml.get('ml_available'): r += f"\n🤖 ML: {ml['ml_signal']}"
        return r

    if any(k in p for k in ["prediksi", "predict", "naik", "turun", "forecast"]):
        lines = [f"🔮 **PREDIKSI ISLM** (Rp {price:,.0f})\n"]
        for k, pred in predictions.items():
            lines.append(f"**{pred['label']}:** Rp {pred['target']:,.0f} ({pred['change_pct']:+.1f}%) {pred['direction']}")
        if supports: lines.append(f"\n🟢 Support: {', '.join(f'Rp {s:,.0f}' for s in supports)}")
        if resistances: lines.append(f"🔴 Resistance: {', '.join(f'Rp {r:,.0f}' for r in resistances)}")
        return "\n".join(lines)

    if any(k in p for k in ["analisa", "teknikal", "technical"]):
        lines = [f"📊 **ANALISA ISLM V4**\n", f"RSI: {rsi:.1f} | MACD: {hist:+.2f} | Stoch: {stoch_k:.1f}"]
        if pro_ta.get('adx'): lines.append(f"ADX: {pro_ta['adx']:.0f} | MFI: {pro_ta.get('mfi', 0):.0f}")
        lines.append(f"\n📢 {sig['label']} | 🏷️ {market_phase}")
        return "\n".join(lines)

    if any(k in p for k in ["whale", "paus"]): return f"🐋 Whale: {whale_ratio*100:.0f}% Buy\n{whale_label}"
    if any(k in p for k in ["berita", "news"]): return f"📰 {f_news}\nSkor: {f_score}/10"
    if any(k in p for k in ["sinyal", "signal"]): return f"📢 {sig['label']} ({sig['confidence']*100:.0f}%)\n{''.join(f'• {r}' + chr(10) for r in sig.get('reasons', []))}"

    return f"🤖 ISLM: Rp {price:,.0f} | {sig['label']} | {sig['trend']}\n\nKetik: harga, prediksi, analisa, whale, sinyal, berita"


# --- ROUTER ---
if not st.session_state.authenticated:
    login_page()
else:
    main_dashboard()
