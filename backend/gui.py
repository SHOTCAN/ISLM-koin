import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import json
import os
import winsound
import sys
import numpy as np
import random
import requests
from datetime import datetime, timedelta
from config import Config
from api import IndodaxAPI

# Configuration
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("green")

HISTORY_FILE = "islm_history.json"
BTC_HISTORY_FILE = "btc_history.json"
SESSION_FILE = "user_session.json"

# --- SECURITY INTEGRATION ---
SECURITY_ENABLED = False
DETECTION_ENGINE = None

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from detection_engine import DetectionEngine, ThreatLevel
    SECURITY_ENABLED = True
    print("✅ DETECTION ENGINE LOADED")
except ImportError:
    print("⚠️ DETECTION ENGINE NOT FOUND - Running in Lite Mode")

from backend.core_logic import MarketProjector, CandleSniper, NewsEngine, TelegramNotifier, FundamentalEngine, QuantAnalyzer

# Removed inline classes (MarketProjector, CandleSniper, NewsEngine, TelegramNotifier, FundamentalEngine, QuantAnalyzer) 
# to use shared backend.core_logic for Web App compatibility.   

class PredictionWindow(ctk.CTkToplevel):
    def __init__(self, parent, horizon_name, horizon_minutes, is_btc=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.horizon_name = horizon_name
        self.horizon_minutes = horizon_minutes
        self.is_btc = is_btc
        coin_name = "BTC" if is_btc else "ISLM"
        self.geometry("600x600")
        self.title(f"🔮 PREDIKSI MASA DEPAN {coin_name}: {horizon_name}")
        self.attributes("-topmost", True)
        
        self.main_frame = ctk.CTkFrame(self, fg_color="#161b22")
        self.main_frame.pack(fill="both", expand=True)

        self.lbl_status = ctk.CTkLabel(self.main_frame, text=f"⏳ MENGAMBIL DATA {coin_name} & SIMULASI...", font=ctk.CTkFont(size=14, weight="bold"), text_color="#fbbf24")
        self.lbl_status.place(relx=0.5, rely=0.5, anchor="center")
        
        self.progress = ctk.CTkProgressBar(self.main_frame, width=400, mode="indeterminate")
        self.progress.place(relx=0.5, rely=0.55, anchor="center")
        self.progress.start()
        
        threading.Thread(target=self.run_simulation, daemon=True).start()

    def run_simulation(self):
        try:
            current_time = time.time()
            prices = []
            
            if self.is_btc:
                # Optimized Caching Logic
                if self.parent.cached_btc_data and (current_time - self.parent.last_btc_fetch_time < 300): # 5 mins cache
                     prices = [float(c['close']) for c in self.parent.cached_btc_data]
                else:
                     raw_data = self.parent.api.get_kline('btcidr', '15') 
                     if raw_data:
                         prices = [float(c['close']) for c in raw_data]
                         self.parent.cached_btc_data = raw_data
                         self.parent.last_btc_fetch_time = current_time
                     else:
                         # Still fallback to simple if get_kline fails entirely
                         prices = [c['close'] for c in self.parent.btc_candles] 
            else:
                 prices = [c['close'] for c in self.parent.candles]

            if len(prices) < 10:
                self.show_error("Data tidak cukup untuk simulasi.")
                return

            current_price = prices[-1]
            vol = MarketProjector.calculate_volatility(np.array(prices))
            drift = MarketProjector.calculate_drift(np.array(prices))
            drift = drift * 0.5 

            # OPTIMIZATION: Reduced Simulations from 2000 to 1000 for speed
            sim_count = 1000 
            paths, _ = MarketProjector.run_monte_carlo(current_price, vol, drift, self.horizon_minutes, sim_count)
            final_prices = paths[:, -1]
            
            p95 = np.percentile(final_prices, 95) 
            p50 = np.percentile(final_prices, 50) 
            p05 = np.percentile(final_prices, 5)  
            
            self.after(0, lambda: self.show_results(paths, p95, p50, p05))
            
        except Exception as e:
            self.after(0, lambda: self.show_error(f"Error: {e}"))

    def show_error(self, msg):
        self.lbl_status.configure(text=msg, text_color="red")
        self.progress.stop()
        self.progress.place_forget()

    def show_results(self, paths, p95, p50, p05):
        self.lbl_status.place_forget()
        self.progress.stop()
        self.progress.place_forget()
        coin = "BTC" if self.is_btc else "ISLM"
        info_frame = ctk.CTkFrame(self.main_frame, fg_color="#0d1117", corner_radius=10)
        info_frame.pack(fill="x", padx=20, pady=20)
        ctk.CTkLabel(info_frame, text=f"TARGET HARGA {coin} ({self.horizon_name})", font=ctk.CTkFont(size=16, weight="bold"), text_color="#38bdf8").pack(pady=(10, 5))
        grid = ctk.CTkFrame(info_frame, fg_color="transparent")
        grid.pack(pady=10)
        self.create_stat(grid, "OPTIMIS (Bull) 🐂", f"Rp {p95:,.0f}", "#4ade80", 0, 0)
        self.create_stat(grid, "PALING MUNGKIN 🎯", f"Rp {p50:,.0f}", "#fbbf24", 0, 1)
        self.create_stat(grid, "PESIMIS (Bear) 🐻", f"Rp {p05:,.0f}", "#f87171", 0, 2)
        chart_frame = ctk.CTkFrame(self.main_frame, fg_color="#0d1117")
        chart_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        fig = Figure(figsize=(5, 3), dpi=100)
        fig.patch.set_facecolor('#0d1117')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d1117')
        indices = np.random.choice(paths.shape[0], 50, replace=False)
        for i in indices:
            ax.plot(paths[i], color='#f59e0b' if self.is_btc else '#3b82f6', alpha=0.1, linewidth=0.5)
        mean_path = np.mean(paths, axis=0)
        ax.plot(mean_path, color='#fbbf24', linewidth=2, label='Rata-Rata')
        upper_bound = np.percentile(paths, 95, axis=0)
        lower_bound = np.percentile(paths, 5, axis=0)
        ax.plot(upper_bound, color='#4ade80', linestyle='--', linewidth=1, alpha=0.7)
        ax.plot(lower_bound, color='#f87171', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(f"Simulasi Monte Carlo {coin} ({self.horizon_name})", color="white", fontsize=10)
        ax.tick_params(axis='x', colors='#888')
        ax.tick_params(axis='y', colors='#888')
        ax.grid(True, linestyle='--', alpha=0.1)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def create_stat(self, parent, title, value, color, r, c):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.grid(row=r, column=c, padx=15)
        ctk.CTkLabel(f, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color="gray").pack()
        ctk.CTkLabel(f, text=value, font=ctk.CTkFont(size=18, weight="bold"), text_color=color).pack()

class PortfolioWindow(ctk.CTkToplevel):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = parent
        self.geometry("400x600")
        self.title("Portofolio Saya 💼")
        self.attributes("-topmost", True)
        self.balance = 0
        self.price = 0
        self.btc_price = 1 
        self.candles = parent.candles
        self.main_frame = ctk.CTkFrame(self, fg_color="#161b22")
        self.main_frame.pack(fill="both", expand=True)
        ctk.CTkLabel(self.main_frame, text="Estimasi Nilai Aset", font=ctk.CTkFont(size=12), text_color="gray").pack(pady=(20, 5))
        self.lbl_value_idr = ctk.CTkLabel(self.main_frame, text="Rp 0", font=ctk.CTkFont(size=32, weight="bold"), text_color="white")
        self.lbl_value_idr.pack()
        self.lbl_value_btc = ctk.CTkLabel(self.main_frame, text="≈ 0.00000000 BTC", font=ctk.CTkFont(size=12), text_color="gray")
        self.lbl_value_btc.pack(pady=(0, 20))
        stats_frame = ctk.CTkFrame(self.main_frame, fg_color="#0d1117", corner_radius=10)
        stats_frame.pack(fill="x", padx=20, pady=10)
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(stats_frame, text="Return (24H)", font=ctk.CTkFont(size=10), text_color="gray").grid(row=0, column=0, pady=(10, 0))
        self.lbl_return_24h = ctk.CTkLabel(stats_frame, text="0.00%", font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        self.lbl_return_24h.grid(row=1, column=0, pady=(0, 10))
        ctk.CTkLabel(stats_frame, text="Return (Sesi)", font=ctk.CTkFont(size=10), text_color="gray").grid(row=0, column=1, pady=(10, 0))
        self.lbl_return_session = ctk.CTkLabel(stats_frame, text="0.00%", font=ctk.CTkFont(size=16, weight="bold"), text_color="white")
        self.lbl_return_session.grid(row=1, column=1, pady=(0, 10))
        ctk.CTkLabel(self.main_frame, text="Pertumbuhan Nilai Aset", font=ctk.CTkFont(size=12, weight="bold"), text_color="white").pack(anchor="w", padx=20, pady=(20, 5))
        self.chart_frame = ctk.CTkFrame(self.main_frame, fg_color="#0d1117", height=200)
        self.chart_frame.pack(fill="x", padx=20)
        self.update_data()
        self.draw_chart()

    def update_data(self):
        try:
             bal_res = self.parent.api.get_balance()
             ticker_res = self.parent.api.get_price('islmidr')
             btc_res = self.parent.api.get_price('btcidr')
             if bal_res.get('success'): self.balance = bal_res['islm'] + bal_res['islm_hold']
             if ticker_res.get('success'): self.price = ticker_res['last']
             if btc_res.get('success'): self.btc_price = btc_res['last']
             val_idr = self.balance * self.price
             val_btc = val_idr / self.btc_price if self.btc_price > 0 else 0
             self.lbl_value_idr.configure(text=f"Rp {val_idr:,.0f}")
             self.lbl_value_btc.configure(text=f"≈ {val_btc:.8f} BTC")
             if len(self.candles) > 0:
                 oldest_price = self.candles[0]['close']
                 change_pct = ((self.price - oldest_price) / oldest_price) * 100
                 color = "#4ade80" if change_pct >= 0 else "#f87171"
                 sign = "+" if change_pct >= 0 else ""
                 self.lbl_return_24h.configure(text=f"{sign}{change_pct:.2f}%", text_color=color)
             if self.parent.session_start_value:
                 start_val = self.parent.session_start_value
                 sess_metrics = ((val_idr - start_val) / start_val) * 100
                 color = "#4ade80" if sess_metrics >= 0 else "#f87171"
                 sign = "+" if sess_metrics >= 0 else ""
                 self.lbl_return_session.configure(text=f"{sign}{sess_metrics:.2f}%", text_color=color)
        except Exception as e: print(f"Portfolio Error: {e}")

    def draw_chart(self):
        if not self.candles: return
        prices = [c['close'] for c in self.candles]
        values = [p * self.balance for p in prices]
        fig = Figure(figsize=(4, 2), dpi=100)
        fig.patch.set_facecolor('#0d1117')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0d1117')
        ax.plot(values, color='#3b82f6', linewidth=2)
        ax.fill_between(range(len(values)), values, min(values), color='#3b82f6', alpha=0.1)
        ax.axis('off') 
        ax.grid(False)
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

class GlossaryWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("600x600")
        self.title("Kamus Trading AI 📚🎓")
        self.attributes("-topmost", True)
        self.scroll = ctk.CTkScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=10, pady=10)
        title = ctk.CTkLabel(self.scroll, text="KAMUS INTELEJEN HAQQ", font=ctk.CTkFont(size=20, weight="bold"), text_color="#4ade80")
        title.pack(pady=(0, 20))
        terms = [
            ("HAMMER 🔨", "Pola Lilin ekor bawah panjang. Bullish Reversal."),
            ("SHOOTING STAR 🌠", "Pola Lilin ekor atas panjang. Bearish Reversal."),
            ("LINEAR REGRESSION 📐", "Garis matematis yang menghitung kemiringan tren (Slope) rata-rata dari 20 data terakhir."),
            ("DIVERGENCE ⚡", "Ketidaksesuaian antara Harga dan Indikator RSI, sinyal pembalikan arah akurat."),
            ("MACD 📊", "Momentum."), ("StochRSI ⚡", "Jenuh Jual/Beli.")
        ]
        for term, desc in terms:
            f = ctk.CTkFrame(self.scroll, fg_color="#161b22", corner_radius=8)
            f.pack(fill="x", pady=5)
            ctk.CTkLabel(f, text=term, font=ctk.CTkFont(size=14, weight="bold"), text_color="#fbbf24").pack(anchor="w", padx=10, pady=(10, 2))
            ctk.CTkLabel(f, text=desc, font=ctk.CTkFont(size=12), text_color="#ddd", wraplength=520, justify="left").pack(anchor="w", padx=10, pady=(0, 10))

class QuantAnalyzer:
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1: return 50
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        if down == 0: return 100
        rs = up / down
        rsi = 100. - 100. / (1. + rs)
        return rsi
    @staticmethod
    def calculate_stoch_rsi(prices, period=14):
        if len(prices) < period: return 50, 50
        recent = prices[-period:]
        highest = np.max(recent)
        lowest = np.min(recent)
        current = prices[-1]
        if highest == lowest: return 50, 50
        stoch = ((current - lowest) / (highest - lowest)) * 100
        return stoch, stoch 
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow: return 0, 0, 0
        ema_fast = QuantAnalyzer.calculate_ema(prices, fast)
        ema_slow = QuantAnalyzer.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line * 0.8 
        hist = macd_line - signal_line
        return macd_line, signal_line, hist
    @staticmethod
    def calculate_fibonacci(high, low):
        diff = high - low
        return {'0.618': high - (diff * 0.618)}
    @staticmethod
    def calculate_sma(prices, window):
        if len(prices) < window: return None
        return np.mean(prices[-window:])
    @staticmethod
    def calculate_ema(prices, window):
        if len(prices) < window: return None
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(prices[-window:], weights, mode='full')[:len(prices[-window:])]
        return a[-1] 
    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        if len(prices) < window: return None, None, None
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return sma + (std * num_std), sma, sma - (std * num_std)
    @staticmethod
    def calculate_atr(highs, lows, closes, period=14):
        if len(highs) < period: return None
        tr = []
        for i in range(1, len(highs)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            tr.append(max(hl, hc, lc))
        if len(tr) < period: return None
        return np.mean(tr[-period:])
    @staticmethod
    def detect_market_phase(prices, vol_proxy): 
        if len(prices) < 20: return "NETRAL"
        sma20 = np.mean(prices[-20:])
        current = prices[-1]
        if abs(current - sma20)/sma20 < 0.01: return "AKUMULASI 🧱"
        elif current > sma20 * 1.02: return "MARKUP (NAIK) 🚀"
        elif current < sma20 * 0.98: return "MARKDOWN (TURUN) 🔻"
        return "KONSOLIDASI ⚖️"
    @staticmethod
    def get_psychological_level(price):
        levels = [150, 200, 225, 250, 275, 300, 350, 400, 500, 750, 1000]
        closest = min(levels, key=lambda x: abs(x - price))
        if abs(price - closest) / price < 0.02: return closest
        return None

class ISLMApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ISLM Monitor - HAQQ NETWORK SPECIALIST 🌙🏰")
        self.geometry("1400x980")
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self.api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)
        self.running = True
        
        self.candles = [] 
        self.btc_candles = [] 
        self.cached_btc_data = [] # New Cache
        self.last_btc_fetch_time = 0
        
        self.current_candle = None
        self.current_candle_start_time = None
        self.candle_interval = 60
        
        self.last_ai_status = None
        self.last_phase = "NETRAL"
        self.last_btc_price = 0
        self.last_chat_time = 0
        self.last_summary_time = time.time()
        self.last_news_time = time.time()
        self.pivot_levels = {'r1': 0, 's1': 0, 'p': 0}
        self.last_whale_alert = 0
        self.last_psy_alert = {'level': 0, 'time': 0}
        self.session_start_value = None
        self.targets = {'entry': 0, 'tp': 0, 'sl': 0}
        self.current_prob_up = 50.0 
        self.whale_power_ratio = 0.5 
        self.ma_cross_state = "neutral" 
        self.toplevel_window = None
        self.portfolio_window = None 
        self.security_engine = None 
        self.last_pattern_alert = "" 
        self.session_data = {}
        self.welcome_msg_shown = False
        self.last_tg_update_id = 0
        self.fundamental_score = 0
        self.fundamental_news = "Menunggu Analisa..."
        
        # Telegram Init
        self.tg_bot = TelegramNotifier(Config.TELEGRAM_TOKEN)
        
        self.load_history()
        self.load_session()
        
        # Load Chat ID from session if available
        if self.session_data.get('telegram_chat_id'):
            self.tg_bot.chat_id = self.session_data['telegram_chat_id']
            
        # Start Telegram Listener
        threading.Thread(target=self.telegram_listener, daemon=True).start()

        # UI
        self.header_frame = ctk.CTkFrame(self, corner_radius=0, height=60, fg_color="#0f1115")
        self.header_frame.grid(row=0, column=0, sticky="ew")
        
        ctk.CTkLabel(self.header_frame, text="🌙 ISLM Haqq Specialist", font=ctk.CTkFont(size=22, weight="bold")).pack(side="left", padx=20, pady=15)
        self.lbl_security = ctk.CTkLabel(self.header_frame, text="🛡️ SYSTEM SECURE", text_color="#4ade80", font=ctk.CTkFont(size=10, weight="bold"))
        self.lbl_security.pack(side="left", padx=10)
        self.status_label = ctk.CTkLabel(self.header_frame, text="Sistem Siap", text_color="gray", font=ctk.CTkFont(size=11))
        self.status_label.pack(side="right", padx=20)

        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # CHANGED: Sidebar is now Scrollable to fix layout issues
        self.sidebar = ctk.CTkScrollableFrame(self.main_frame, width=340, fg_color="#161b22", corner_radius=10)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        # self.sidebar.grid_propagate(False) # Removed for Scrollable

        ctk.CTkLabel(self.sidebar, text="📊 MARKET PHASE", font=ctk.CTkFont(size=14, weight="bold"), text_color="#fbbf24").pack(pady=(10, 5))
        self.dashboard_frame = ctk.CTkFrame(self.sidebar, fg_color="#0d1117", corner_radius=8)
        self.dashboard_frame.pack(fill="x", padx=10, pady=5)
        self.lbl_phase = ctk.CTkLabel(self.dashboard_frame, text="Fase: MENGANALISA...", font=ctk.CTkFont(size=12, weight="bold"), text_color="#38bdf8")
        self.lbl_phase.pack(anchor="w", padx=10, pady=5)
        self.lbl_pnl = ctk.CTkLabel(self.dashboard_frame, text="Session PnL: Rp 0 (0%)", font=ctk.CTkFont(size=11), text_color="#4ade80")
        self.lbl_pnl.pack(anchor="w", padx=10, pady=5)
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#333").pack(fill="x", padx=10, pady=10)
        self.lbl_price = self.create_sidebar_stat(self.sidebar, "HARGA ISLM (IDR)", "Rp --", 26)
        self.lbl_btc = self.create_sidebar_stat(self.sidebar, "GLOBAL (BTC)", "Rp --", 14)
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#333").pack(fill="x", padx=10, pady=10)
        self.lbl_bal = self.create_sidebar_stat(self.sidebar, "SALDO (ISLM)", "--", 18)
        self.lbl_port = self.create_sidebar_stat(self.sidebar, "NILAI", "Rp --", 18)
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#333").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="🐋 Komunitas vs Smart Money", font=ctk.CTkFont(size=10, weight="bold"), text_color="gray").pack()
        self.power_bar = ctk.CTkProgressBar(self.sidebar, width=280, height=12)
        self.power_bar.pack(pady=5)
        self.power_bar.set(0.5)
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#333").pack(fill="x", padx=10, pady=10)
        
        # PREDICTION BUTTONS ISLM
        ctk.CTkLabel(self.sidebar, text="PREDIKSI ISLM (IDR)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#c084fc").pack(pady=(5, 5))
        pred_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        pred_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(pred_frame, text="🔮 1 HARI", width=100, fg_color="#581c87", hover_color="#6b21a8", 
                      command=lambda: self.open_prediction("1 HARI", 1440, False)).pack(side="left", padx=(0, 5), expand=True)
        ctk.CTkButton(pred_frame, text="📆 1 MINGGU", width=100, fg_color="#4c1d95", hover_color="#5b21b6",
                      command=lambda: self.open_prediction("1 MINGGU", 10080, False)).pack(side="right", padx=(5, 0), expand=True)
        
        # PREDICTION BUTTONS BTC (NEW)
        ctk.CTkLabel(self.sidebar, text="PREDIKSI BITCOIN (IDR)", font=ctk.CTkFont(size=12, weight="bold"), text_color="#f59e0b").pack(pady=(15, 5))
        btc_pred_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btc_pred_frame.pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(btc_pred_frame, text="₿ 1 HARI", width=100, fg_color="#b45309", hover_color="#d97706", 
                      command=lambda: self.open_prediction("1 HARI", 1440, True)).pack(side="left", padx=(0, 5), expand=True)
        ctk.CTkButton(btc_pred_frame, text="₿ 1 MINGGU", width=100, fg_color="#92400e", hover_color="#b45309",
                      command=lambda: self.open_prediction("1 MINGGU", 10080, True)).pack(side="right", padx=(5, 0), expand=True)
        
        ctk.CTkFrame(self.sidebar, height=1, fg_color="#333").pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(self.sidebar, text="🤖 HAQQ INTELLIGENCE", font=ctk.CTkFont(size=12, weight="bold"), text_color="#4ade80").pack(pady=(5, 5))
        self.chat_log = ctk.CTkTextbox(self.sidebar, height=250, fg_color="#0d1117", text_color="#ddd", font=ctk.CTkFont(size=11))
        self.chat_log.pack(fill="both", expand=True, padx=10, pady=10)
        self.chat_log.insert("end", "🤖: Spesialis ISLM Aktif. Memantau Jaringan Haqq...\n\n")
        self.chat_log.configure(state="disabled")
        
        self.chat_log._textbox.tag_config("bull", foreground="#4ade80")
        self.chat_log._textbox.tag_config("bear", foreground="#f87171")
        self.chat_log._textbox.tag_config("warn", foreground="#fbbf24")
        self.chat_log._textbox.tag_config("info", foreground="#94a3b8")
        self.chat_log._textbox.tag_config("math", foreground="#c084fc")
        self.chat_log._textbox.tag_config("whale", foreground="#38bdf8")
        self.chat_log._textbox.tag_config("news", foreground="#ffffff") 
        self.chat_log._textbox.tag_config("sniper", foreground="#e879f9") 

        btn_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=10, pady=10)
        self.btn_up = ctk.CTkButton(btn_frame, text="🚀 Akan Naik?", fg_color="#10b981", hover_color="#059669", 
                                    text_color="white", font=ctk.CTkFont(weight="bold"),
                                    command=lambda: self.ask_ai("UP"))
        self.btn_up.pack(side="left", expand=True, padx=(0, 5))
        self.btn_down = ctk.CTkButton(btn_frame, text="🔻 Akan Turun?", fg_color="#ef4444", hover_color="#dc2626", 
                                      text_color="white", font=ctk.CTkFont(weight="bold"),
                                      command=lambda: self.ask_ai("DOWN"))
        self.btn_down.pack(side="right", expand=True, padx=(5, 0))

        action_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        action_frame.pack(fill="x", padx=10, pady=(5, 20)) # Increased bottom padding for scroll area
        self.btn_kamus = ctk.CTkButton(action_frame, text="📚 Kamus AI", fg_color="#3b82f6", hover_color="#2563eb",
                                       text_color="white", width=140, command=self.open_glossary)
        self.btn_kamus.pack(side="left", padx=(0, 5))
        self.btn_port = ctk.CTkButton(action_frame, text="💼 Portofolio", fg_color="#8b5cf6", hover_color="#7c3aed",
                                      text_color="white", width=140, command=self.open_portfolio)
        self.btn_port.pack(side="right", padx=(5, 0))

        # TELEGRAM CONNECT BUTTON
        self.tg_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.tg_frame.pack(fill="x", padx=10, pady=(5, 20))
        self.btn_tg = ctk.CTkButton(self.tg_frame, text="🔗 Connect Telegram", fg_color="#22d3ee", hover_color="#0891b2",
                                    text_color="black", command=self.connect_telegram)
        self.btn_tg.pack(fill="x")
        
        if self.tg_bot.chat_id:
            self.btn_tg.configure(text="✅ Telegram Connected", state="disabled", fg_color="#064e3b", text_color="white")

        self.chart_bg = ctk.CTkFrame(self.main_frame, fg_color="#161b22", corner_radius=10)
        self.chart_bg.grid(row=0, column=1, sticky="nsew")
        
        self.alert_overlay = ctk.CTkFrame(self, fg_color="#7f1d1d", corner_radius=0)
        ctk.CTkLabel(self.alert_overlay, text="⚠️ CRASH DETECTED ⚠️", text_color="white", font=ctk.CTkFont(size=24, weight="bold")).pack(expand=True)

        self.init_chart()
        self.update_thread = threading.Thread(target=self.auto_refresh_loop, daemon=True)
        self.update_thread.start()
        
        if SECURITY_ENABLED:
            self.security_engine = DetectionEngine()
            threading.Thread(target=self.perform_security_scan, daemon=True).start()
        else:
             self.lbl_security.configure(text="⚠️ SECURITY OFF", text_color="gray")

    def perform_security_scan(self):
        try:
             self.add_chat_message("🛡️ MEMULAI PEMINDAIAN KEAMANAN...", "info")
             time.sleep(1) 
             files = ["api.py", "config.py", __file__]
             threats_found = 0
             for f in files:
                 if os.path.exists(f):
                     threats = self.security_engine.scan_file(f)
                     if threats: threats_found += len(threats)
             if threats_found == 0:
                 self.add_chat_message("✅ SCAN SELESAI: Sistem Bersih (Clean) & Aman dari Virus.", "bull")
                 self.lbl_security.configure(text="🛡️ SYSTEM SECURE", text_color="#4ade80")
             else:
                 self.add_chat_message(f"⚠️ PERINGATAN: Ditemukan {threats_found} Anomali File!", "bear")
                 self.lbl_security.configure(text="⚠️ THREAT DETECTED", text_color="#f87171")
        except Exception as e: print(e)

    def create_sidebar_stat(self, parent, title, value, val_size=20):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=15, pady=3)
        ctk.CTkLabel(frame, text=title, text_color="gray", font=ctk.CTkFont(size=9, weight="bold")).pack(anchor="w")
        lbl = ctk.CTkLabel(frame, text=value, font=ctk.CTkFont(size=val_size, weight="bold"))
        lbl.pack(anchor="w")
        return lbl

    def add_chat_message(self, message, tag="info"):
        timestamp = datetime.now().strftime("%H:%M:%S") 
        full_msg = f"[{timestamp}] 🤖: {message}\n\n"
        self.chat_log.configure(state="normal")
        self.chat_log._textbox.insert("end", full_msg, tag)
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def ask_ai(self, direction):
        prob = self.current_prob_up
        if direction == "UP":
            if prob >= 65:
                msg = f"🚀 Analisa Naik (Probabilitas {prob:.1f}%): POTENSI KUAT! Market sedang bergairah (Markup)."
                self.add_chat_message(msg, "bull")
            elif prob >= 50:
                msg = f"📈 Analisa Naik (Probabilitas {prob:.1f}%): Cukup Positif. Ada tenaga beli, tapi masih berjuang menembus Resistance."
                self.add_chat_message(msg, "math")
            else:
                msg = f"⚠️ Analisa Naik (Probabilitas {prob:.1f}%): Berat Bos. Tekanan jual masih dominan."
                self.add_chat_message(msg, "warn")
        elif direction == "DOWN":
            if prob <= 35:
                msg = f"🔻 Analisa Turun (Risiko {100-prob:.1f}%): BAHAYA! Sinyal jual sangat kuat. Hati-hati longsor."
                self.add_chat_message(msg, "bear")
            elif prob <= 50:
                msg = f"🛡️ Analisa Turun (Risiko {100-prob:.1f}%): Ada risiko koreksi wajar (Konsolidasi)."
                self.add_chat_message(msg, "warn")
            else:
                msg = f"🧱 Analisa Turun: Kecil kemungkinan dump."
                self.add_chat_message(msg, "bull")

    def open_glossary(self):
        if self.toplevel_window is None or not self.toplevel_window.winfo_exists():
            self.toplevel_window = GlossaryWindow(self)
        try: self.toplevel_window.focus()
        except: pass

    def open_portfolio(self):
        if self.portfolio_window is None or not self.portfolio_window.winfo_exists():
            self.portfolio_window = PortfolioWindow(self)
        try: self.portfolio_window.focus()
        except: pass

    def open_prediction(self, name, minutes, is_btc):
        coin = "BTC" if is_btc else "ISLM"
        self.add_chat_message(f"🔮 MENGHITUNG MASA DEPAN {coin} ({name})... Harap tunggu...", "math")
        win = PredictionWindow(self, name, minutes, is_btc)
        try: win.focus()
        except: pass

    def play_sound(self, sound_type):
        try:
            if sound_type == "alert": winsound.Beep(1000, 500)
            elif sound_type == "notify": winsound.Beep(800, 200)
            elif sound_type == "whale": winsound.Beep(600, 800)
        except: pass

    def update_ai_analysis(self, price, high, low, balance, btc_price, depth_data):
        pivot = (high + low + price) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        self.pivot_levels = {'p': pivot, 'r1': r1, 's1': s1}
        prices = [c['close'] for c in self.candles] + [price]
        highs = [c['high'] for c in self.candles] + [high]
        lows_arr = [c['low'] for c in self.candles] + [low]
        if len(prices) < 20: return

        # CALCULATIONS
        rsi = QuantAnalyzer.calculate_rsi(np.array(prices))
        stoch_k, stoch_d = QuantAnalyzer.calculate_stoch_rsi(np.array(prices)) 
        macd_line, macd_signal, macd_hist = QuantAnalyzer.calculate_macd(np.array(prices)) 
        slope = MarketProjector.calculate_long_term_trend(prices)
        divergence = MarketProjector.calculate_rsi_divergence(prices, [p for p in prices]) 
 
        ma7 = QuantAnalyzer.calculate_sma(prices, 7)
        ema25 = QuantAnalyzer.calculate_ema(prices, 25)
        ema99 = QuantAnalyzer.calculate_ema(prices, 99)
        phase = QuantAnalyzer.detect_market_phase(prices, None)
        self.lbl_phase.configure(text=f"Fase: {phase}")

        if phase != self.last_phase:
            if phase == "KONSOLIDASI ⚖️":
                self.add_chat_message(f"⚖️ INFO FASE: Masuk KONSOLIDASI. Harga istirahat.", "info")
            else:
                self.add_chat_message(f"🔄 PERUBAHAN FASE: {phase}.", "math")
            self.last_phase = phase

        patterns = CandleSniper.analyze_patterns(self.candles + [self.current_candle])
        if patterns:
             pattern_str = ", ".join(patterns)
             if pattern_str != self.last_pattern_alert:
                 self.last_pattern_alert = pattern_str
                 msg = f"🎯 SNIPER ALERT: Pola {pattern_str} terdeteksi! Waspada Reversal."
                 self.add_chat_message(msg, "sniper")
                 self.tg_bot.send_message(f"🎯 *SNIPER ALERT*\nPola: {pattern_str}\nHarga: {rp(price)}")

        buy_vol, sell_vol = 0, 0
        if depth_data:
            buys = depth_data.get('buy', [])
            sells = depth_data.get('sell', [])
            buy_vol = sum([float(x[1]) for x in buys[:50]])
            sell_vol = sum([float(x[1]) for x in sells[:50]])
            total_vol = buy_vol + sell_vol
            if total_vol > 0:
                self.whale_power_ratio = buy_vol / total_vol
                self.power_bar.set(self.whale_power_ratio)

        # SCORE LOGIC
        prob_up = 50.0 
        
        # 0. Global Market Context (BTC)
        if self.last_btc_price > 0 and btc_price > 0:
            if btc_price > self.last_btc_price: prob_up += 3
            elif btc_price < self.last_btc_price: prob_up -= 3
            
        # 1. Fundamental Score (Islamic Finance)
        prob_up += (self.fundamental_score * 2) # Impact -6 to +10

        if buy_vol > (sell_vol * 1.2): prob_up += 15
        elif sell_vol > (buy_vol * 1.2): prob_up -= 15
        if phase == "MARKUP (NAIK) 🚀": prob_up += 15
        elif phase == "MARKDOWN (TURUN) 🔻": prob_up -= 15
        
        if ma7 and ema25 and ema99:
            if price > ma7 and ma7 > ema25 and ema25 > ema99: prob_up += 20
            elif price < ma7 and ma7 < ema25 and ema25 < ema99: prob_up -= 20
        
        if macd_hist > 0: prob_up += 5
        elif macd_hist < 0: prob_up -= 5
        if stoch_k < 20: prob_up += 10 
        elif stoch_k > 80: prob_up -= 10 

        # New Math Score
        if slope > 0.5: prob_up += 10
        elif slope < -0.5: prob_up -= 10
        
        if prob_up == 50.0:
            if buy_vol > sell_vol: prob_up = 52.0
            else: prob_up = 48.0
        prob_up = max(0, min(100, prob_up))
        self.current_prob_up = prob_up

        current_time = time.time()
        
        if stoch_k < 20 and self.current_prob_up > 40:
             if random.random() < 0.2: 
                 self.add_chat_message(f"⚡ STOCH RSI UPDATE: Oversold ({stoch_k:.1f}). Potensi pantulan naik!", "math")
        
        if (current_time - self.last_news_time > 120 + random.randint(0, 120)):
            self.last_news_time = current_time
            news = NewsEngine.generate_news(phase, 0, self.whale_power_ratio)
            self.add_chat_message(f"📰 BERITA JARINGAN: {news}", "news")

        if (current_time - self.last_summary_time > 60):
            self.last_summary_time = current_time
            direction = "NETRAL ⏸️"
            reason = "Menunggu momentum."
            if prob_up >= 65:
                direction = "POTENSI NAIK 🚀"
                reason = "Trend Slope: {slope:.2f} (Positif) & Volume Beli."
            elif prob_up <= 35:
                direction = "POTENSI TURUN 🔻"
                reason = "Trend Slope: {slope:.2f} (Negatif) & Tekanan Jual."
            btc_ctx = "BTC Stabil"
            if btc_price > self.last_btc_price * 1.0005: btc_ctx = "BTC Hijau"
            elif btc_price < self.last_btc_price * 0.9995: btc_ctx = "BTC Merah"

            msg = f"⏱️ KESIMPULAN MENIT INI: {direction}\nAlasan: {reason} (Prob: {prob_up:.1f}%)\n[Global: {btc_ctx} | Slope: {slope:.2f}]"
            self.add_chat_message(msg, "math")

        new_status = "NEUTRAL"
        if prob_up >= 75:
            new_status = "STRONG_BUY"
            if self.last_ai_status != "STRONG_BUY":
                self.add_chat_message(f"🚀 HAQQ BULLISH! Probabilitas {prob_up:.0f}%.", "bull")
                self.play_sound("notify")
        elif prob_up <= 25:
            new_status = "STRONG_SELL"
            if self.last_ai_status != "STRONG_SELL":
                self.add_chat_message(f"⚠️ HATI-HATI! Tekanan Jual Tinggi ({100-prob_up:.0f}%).", "bear")
                self.play_sound("alert")
                self.tg_bot.send_message(f"⚠️ *STRONG SELL SIGNAL*\nProbabilitas Turun: {100-prob_up:.0f}%\nHarga: {rp(price)}")

        self.last_ai_status = new_status
        self.last_btc_price = btc_price

        if prob_up < 20:
             self.alert_overlay.place(relx=0, rely=0, relwidth=1, relheight=0.08)
             self.play_sound("alert")
        else:
             self.alert_overlay.place_forget()

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    for c in data: c['time'] = datetime.fromisoformat(c['time'])
                    self.candles = data
            except: pass

    def save_history(self):
        try:
            data = [{'time': c['time'].isoformat(), **{k: v for k, v in c.items() if k != 'time'}} for c in self.candles]
            with open(HISTORY_FILE, 'w') as f: json.dump(data, f)
        except: pass

    def load_session(self):
        if os.path.exists(SESSION_FILE):
            try:
                with open(SESSION_FILE, 'r') as f:
                    self.session_data = json.load(f)
            except: pass

    def save_session(self):
        try:
            with open(SESSION_FILE, 'w') as f:
                json.dump(self.session_data, f)
        except: pass

    def init_chart(self):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.fig.patch.set_facecolor('#161b22')
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_vol = self.fig.add_subplot(gs[1], sharex=self.ax)
        self.ax.set_facecolor('#161b22')
        self.ax_vol.set_facecolor('#161b22')
        for ax in [self.ax, self.ax_vol]:
            for spine in ax.spines.values(): spine.set_color('#333')
            ax.tick_params(axis='x', colors='#888')
            ax.tick_params(axis='y', colors='#888')
            ax.grid(True, linestyle='--', alpha=0.1, color='#666')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_bg)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def process_price_update(self, price):
        now = datetime.now()
        if self.current_candle is None:
            self.current_candle_start_time = now
            self.current_candle = {'time': now, 'open': price, 'high': price, 'low': price, 'close': price}
        else:
            time_diff = (now - self.current_candle_start_time).total_seconds()
            if time_diff >= self.candle_interval:
                self.candles.append(self.current_candle.copy())
                if len(self.candles) > 100: self.candles.pop(0)
                self.save_history()
                self.session_data['last_price'] = price
                self.save_session()
                self.current_candle_start_time = now
                self.current_candle = {'time': now, 'open': price, 'high': price, 'low': price, 'close': price}
            else:
                self.current_candle['high'] = max(self.current_candle['high'], price)
                self.current_candle['low'] = min(self.current_candle['low'], price)
                self.current_candle['close'] = price

    def draw_candlestick(self):
        if not self.ax or not self.ax_vol: return 

        try:
            self.ax.clear()
            self.ax_vol.clear()
            self.ax.grid(True, linestyle='--', alpha=0.1, color='#666')
            self.ax_vol.grid(True, linestyle='--', alpha=0.1, color='#666')
            
            data_to_plot = self.candles + ([self.current_candle] if self.current_candle else [])
            if not data_to_plot:
                self.ax.text(0.5, 0.5, "Menunggu Data...", color="white", ha="center", transform=self.ax.transAxes)
                self.canvas.draw()
                return

            visible_data = data_to_plot[-50:]
            x_indices = range(len(visible_data))
            opens = [d['open'] for d in visible_data]
            highs = [d['high'] for d in visible_data]
            lows = [d['low'] for d in visible_data]
            closes = [d['close'] for d in visible_data]
            colors = ['#4ade80' if c >= o else '#ef4444' for o, c in zip(opens, closes)]
            
            # Robust Drawing
            self.ax.vlines(x=x_indices, ymin=lows, ymax=highs, color=colors, linewidth=1)
            self.ax.bar(x_indices, [abs(c-o) for o,c in zip(opens, closes)], bottom=[min(o,c) for o,c in zip(opens, closes)], color=colors, width=0.6)

            if len(visible_data) > 7:
                sma7 = [QuantAnalyzer.calculate_sma([d['close'] for d in visible_data[:i+1]], 7) for i in range(len(visible_data))]
                if sma7 and not all(v is None for v in sma7):
                    self.ax.plot(x_indices, sma7, color='#fbbf24', linewidth=1.2, alpha=0.9, label='MA7')
            
            if len(visible_data) > 25:
                ema25 = [QuantAnalyzer.calculate_ema([d['close'] for d in visible_data[:i+1]], 25) for i in range(len(visible_data))]
                if ema25 and not all(v is None for v in ema25):
                    self.ax.plot(x_indices, ema25, color='#f97316', linewidth=1.5, alpha=0.8, label='EMA25')
                    
            if len(visible_data) > 99:
                ema99 = [QuantAnalyzer.calculate_ema([d['close'] for d in visible_data[:i+1]], 99) for i in range(len(visible_data))]
                if ema99 and not all(v is None for v in ema99):
                    self.ax.plot(x_indices, ema99, color='#8b5cf6', linewidth=2.0, alpha=0.9, label='EMA99')

            patterns = CandleSniper.analyze_patterns(visible_data)
            
            if self.pivot_levels['r1'] > 0:
                self.ax.axhline(y=self.pivot_levels['r1'], color='#4ade80', linestyle=':', alpha=0.4)
                self.ax.axhline(y=self.pivot_levels['s1'], color='#f87171', linestyle=':', alpha=0.4)
            if self.targets['tp'] > 0:
                self.ax.axhline(y=self.targets['tp'], color='#22d3ee', linestyle='--', alpha=0.5)
                self.ax.axhline(y=self.targets['sl'], color='#fb7185', linestyle='--', alpha=0.5)

            vol_proxy = [abs(c-o)*100 + 10 for o,c in zip(opens, closes)]
            self.ax_vol.bar(x_indices, vol_proxy, color=colors, alpha=0.3)
            self.ax.set_xlim(-1, len(visible_data))
            self.ax_vol.set_xlim(-1, len(visible_data))
            if lows and highs:
                min_y = min(lows) * 0.998; max_y = max(highs) * 1.002
                self.ax.set_ylim(min_y, max_y)
            
            if data_to_plot:
                self.ax.set_title(f"ISLM/IDR • {data_to_plot[-1]['close']:,.0f}", color="white", loc='left', fontsize=10)
            
            times = [d['time'] for d in visible_data]
            self.ax_vol.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: times[int(x)].strftime('%H:%M') if 0 <= int(x) < len(times) else ''))
            self.ax.tick_params(labelbottom=False)
            self.ax_vol.tick_params(axis='x', labelsize=7)
            self.ax.tick_params(axis='y', labelsize=7)
            self.fig.tight_layout()
            self.canvas.draw()
        except Exception as e:
            pass # Silent fail to prevent crash loop

    def auto_refresh_loop(self):
        while self.running:
            self.fetch_data()
            time.sleep(2)

    def fetch_data(self):
        try:
            bal_resp = self.api.get_balance()
            price_resp = self.api.get_price('islmidr')
            btc_resp = self.api.get_price('btcidr')
            depth_resp = self.api.get_depth('islmidr')
            
            # Background BTC Fetch (Every 30s approx to avoid rate limit)
            if random.random() < 0.2: 
                 threading.Thread(target=self.bg_fetch_btc, daemon=True).start()

            # Periodic Fundamental Update (Every 5 mins roughly)
            if random.random() < 0.05:
                score, news = FundamentalEngine.analyze_market_sentiment()
                self.fundamental_score = score
                self.fundamental_news = news
                
            self.after(0, lambda: self.update_ui(bal_resp, price_resp, btc_resp, depth_resp))
        except Exception as e:
            self.after(0, lambda: self.status_label.configure(text=f"Error: {str(e)[:30]}"))

    def bg_fetch_btc(self):
        try:
            raw_data = self.api.get_kline('btcidr', '15')
            if raw_data:
                self.cached_btc_data = raw_data
                self.last_btc_fetch_time = time.time()
        except: pass

    def update_ui(self, bal_data, ticker_data, btc_data, depth_data):
        now_str = datetime.now().strftime('%H:%M:%S')
        self.status_label.configure(text=f"🟢 Haqq Live | {now_str}")
        fmt = lambda x: f"{float(x):,.2f}"
        rp = lambda x: f"Rp {float(x):,.0f}"
        islm_total = 0
        if bal_data.get('success'):
            islm_total = bal_data['islm'] + bal_data['islm_hold']
            self.lbl_bal.configure(text=fmt(islm_total))
        if btc_data.get('success'): self.lbl_btc.configure(text=rp(btc_data['last']))
        if ticker_data.get('success'):
            price = ticker_data['last']
            self.lbl_price.configure(text=rp(price))
            current_value = islm_total * price
            if bal_data.get('success'):
                self.lbl_port.configure(text=rp(current_value))
                if self.session_start_value is None:
                    self.session_start_value = current_value
                pnl = current_value - self.session_start_value
                pnl_pct = (pnl / self.session_start_value * 100) if self.session_start_value > 0 else 0
                color = "#4ade80" if pnl >= 0 else "#f87171"
                sign = "+" if pnl >= 0 else ""
                self.lbl_pnl.configure(text=f"Session PnL: {sign}Rp {pnl:,.0f} ({sign}{pnl_pct:.2f}%)", text_color=color)
            
            # Welcome Message Logic
            if not self.welcome_msg_shown:
                self.welcome_msg_shown = True
                last_price = self.session_data.get('last_price')
                if last_price:
                    diff = ((price - last_price) / last_price) * 100
                    direction = "NAIK 🚀" if diff >= 0 else "TURUN 🔻"
                    msg = f"👋 WELCOME BACK BOS! Harga terakhir Anda lihat: Rp {last_price:,.0f}.\nSekarang: Rp {price:,.0f} ({direction} {abs(diff):.2f}%).\nSelamat memantau lagi!"
                    self.add_chat_message(msg, "whale")
                    self.tg_bot.send_message(f"👋 *WELCOME BACK*\nLast: Rp {last_price:,.0f}\nNow: Rp {price:,.0f} ({direction} {abs(diff):.2f}%)")
                else:
                    self.add_chat_message(f"👋 SELAMAT DATANG DI ISLM MONITOR!\nHarga saat ini: Rp {price:,.0f}. AI Siap membantu.", "whale")
                    self.tg_bot.send_message(f"👋 *ISLM MONITOR STARTED*\nHarga: Rp {price:,.0f}")
            
            btc_price = btc_data.get('last', 0) if btc_data.get('success') else 0
            self.update_ai_analysis(price, ticker_data['high'], ticker_data['low'], islm_total, btc_price, depth_data)
            self.process_price_update(price)
            self.draw_candlestick()

    def connect_telegram(self):
        self.btn_tg.configure(text="🔍 Searching...", state="disabled")
        self.update()
        
        # Run search in thread
        threading.Thread(target=self._search_telegram_user, daemon=True).start()

    def telegram_listener(self):
        print("🤖 Telegram Listener Active")
        while self.running:
            try:
                if not self.tg_bot.chat_id: 
                    time.sleep(5)
                    continue
                    
                updates = self.tg_bot.get_updates(offset=self.last_tg_update_id + 1)
                for u in updates:
                    self.last_tg_update_id = u["update_id"]
                    if "message" not in u: continue
                    msg = u["message"]
                    text = msg.get("text", "").lower()
                    chat_id = msg["chat"]["id"]
                    
                    # Command Handling
                    if "/start" in text or "/help" in text:
                        reply = (
                            "🤖 *ISLM MONITOR COMMANDS*\n"
                            "/status - Cek Harga & Sinyal AI\n"
                            "/prediksi - Target Harga 1 Hari\n"
                            "/minggu - Target Harga 1 Minggu\n"
                            "/2minggu - Target Harga 14 Hari (Spesial Ramadhan)\n"
                            "/news - Berita Fundamental\n"
                            "/fundamental - Analisa Sentimen Pasar"
                        )
                        self.tg_bot.send_message(reply)
                        
                    elif "/status" in text or "/cek" in text:
                        price = self.session_data.get('last_price', 0)
                        trend = "NAIK 🚀" if self.current_prob_up > 50 else "TURUN 🔻"
                        reply = (
                            f"📊 *STATUS PASAR*\n"
                            f"Harga: Rp {price:,.0f}\n"
                            f"Probabilitas Naik: {self.current_prob_up:.1f}%\n"
                            f"Tren: {trend}\n"
                            f"Fase: {self.last_phase}"
                        )
                        self.tg_bot.send_message(reply)
                        
                    elif "/prediksi" in text or "/1hari" in text:
                         self.tg_bot.send_message("🔮 Sedang menghitung prediksi 1 Hari... (Mohon tunggu)")
                         self.run_tg_prediction(1440, "1 HARI")
                             
                    elif "/minggu" in text:
                         self.tg_bot.send_message("📆 Sedang menghitung prediksi 1 Minggu... (Mohon tunggu)")
                         self.run_tg_prediction(10080, "1 MINGGU")

                    elif "/2minggu" in text or "/puasa" in text:
                         self.tg_bot.send_message("🌙 Sedang menghitung efek Ramadhan (14 Hari)...")
                         self.run_tg_prediction(20160, "2 MINGGU (RAMADHAN EFFECT)")

                    elif "/1menit" in text:
                         self.run_tg_prediction(1, "1 MENIT")
                    
                    elif "/5menit" in text:
                         self.run_tg_prediction(5, "5 MENIT")

                    elif "/news" in text:
                        self.tg_bot.send_message(f"📰 *BERITA FUNDAMENTAL*\n{self.fundamental_news}")

                    elif "/fundamental" in text:
                        score_text = "POSITIF" if self.fundamental_score > 0 else "NEGATIF"
                        self.tg_bot.send_message(f"🧠 *ANALISA FUNDAMENTAL*\nSentimen: {score_text} (Skor: {self.fundamental_score})\nFaktor: {self.fundamental_news}")

            except Exception as e: 
                print(f"TG Error: {e}")
                time.sleep(5)
            time.sleep(2)

    def run_tg_prediction(self, minutes, label):
         if self.candles:
             current_price = self.candles[-1]['close']
             prices = [c['close'] for c in self.candles]
             vol = MarketProjector.calculate_volatility(np.array(prices))
             drift = MarketProjector.calculate_drift(np.array(prices)) * 0.5
             
             # Adjust steps for very short horizon
             steps = 10 if minutes < 10 else 100
             
             paths, _ = MarketProjector.run_monte_carlo(current_price, vol, drift, minutes, 500)
             final = paths[:, -1]
             p50 = np.percentile(final, 50)
             diff = ((p50 - current_price) / current_price) * 100
             sign = "+" if diff >= 0 else ""
             self.tg_bot.send_message(f"🎯 *PREDIKSI {label}*\nTarget: Rp {p50:,.0f} ({sign}{diff:.2f}%)")
         else:
             self.tg_bot.send_message("⚠️ Data belum cukup untuk prediksi.")

    def _search_telegram_user(self):
        for i in range(5):
            updates = self.tg_bot.get_updates()
            for u in updates:
                 if "message" in u:
                     chat_id = u["message"]["chat"]["id"]
                     username = u["message"]["chat"].get("username", "Bos")
                     self.tg_bot.chat_id = chat_id
                     self.session_data['telegram_chat_id'] = chat_id
                     self.save_session()
                     self.tg_bot.send_message(f"✅ *CONNECTED*\nHalo {username}! ISLM Monitor berhasil terhubung.")
                     self.after(0, lambda: self.btn_tg.configure(text=f"✅ Connected: {username}", fg_color="#064e3b", text_color="white"))
                     self.after(0, lambda: self.add_chat_message(f"📱 TELEGRAM CONNECTED: {username}", "bull"))
                     return
            time.sleep(2)
        
        self.after(0, lambda: self.btn_tg.configure(text="❌ Gagal. Chat bot dulu!", state="normal", fg_color="#ef4444"))
        self.after(0, lambda: self.add_chat_message("⚠️ Gagal konek Telegram. Pastikan Bos sudah chat /start ke bot @AI_aploud_automation_bot dulu ya!", "warn"))


if __name__ == "__main__":
    app = ISLMApp()
    app.mainloop()
