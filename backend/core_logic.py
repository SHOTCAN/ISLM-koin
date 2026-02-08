import numpy as np
import random
import requests
import threading
import time
from config import Config

class MarketProjector:
    @staticmethod
    def calculate_volatility(prices): 
        if len(prices) < 2: return 0.001
        pct_changes = np.abs(np.diff(prices) / prices[:-1])
        return np.mean(pct_changes)

    @staticmethod
    def calculate_long_term_trend(prices): 
        if len(prices) < 20: return 0
        x = np.arange(len(prices))
        y = np.array(prices)
        z = np.polyfit(x, y, 1) # Degree 1 (Line)
        return z[0] # Returns slope

    @staticmethod
    def calculate_rsi_divergence(prices, rsi_values, window=10):
        if len(prices) < window or len(rsi_values) < window: return None
        
        price_slice = prices[-window:]
        rsi_slice = rsi_values[-window:]
        
        # Bullish Divergence: Price Lower Low, RSI Higher Low
        if price_slice[-1] < min(price_slice[:-1]) and rsi_slice[-1] > min(rsi_slice[:-1]):
            return "BULL_DIV"
            
        # Bearish Divergence: Price Higher High, RSI Lower High
        if price_slice[-1] > max(price_slice[:-1]) and rsi_slice[-1] < max(rsi_slice[:-1]):
            return "BEAR_DIV"
            
        return None

    @staticmethod
    def calculate_drift(prices):
        if len(prices) < 2: return 0
        pct_changes = np.diff(prices) / prices[:-1]
        return np.mean(pct_changes)

    @staticmethod
    def run_monte_carlo(current_price, volatility, drift, horizon_minutes, simulations=1000):
        # OPTIMIZED: Use fixed steps regardless of horizon to be instant
        steps = 100 
        dt = horizon_minutes / steps 
        
        # Scale drift and vol to step size
        step_drift = drift * dt
        step_vol = volatility * np.sqrt(dt)
        
        shocks = np.random.normal(0, 1, (simulations, steps))
        daily_returns = 1 + step_drift + (step_vol * shocks)
        price_paths = np.cumprod(daily_returns, axis=1) * current_price
        start_col = np.full((simulations, 1), current_price)
        price_paths = np.hstack((start_col, price_paths))
        return price_paths

class CandleSniper:
    @staticmethod
    def analyze_patterns(candles):
        if len(candles) < 3: return []
        patterns = []
        try:
            c = candles[-1]
            prev = candles[-2]
            
            open_p, close_p = c['open'], c['close']
            high_p, low_p = c['high'], c['low']
            
            body_size = abs(close_p - open_p)
            upper_wick = high_p - max(close_p, open_p)
            lower_wick = min(close_p, open_p) - low_p
            
            is_bullish = close_p > open_p
            is_bearish = close_p < open_p
            
            if lower_wick > (body_size * 2) and upper_wick < (body_size * 0.5):
                patterns.append("HAMMER 🔨") if is_bullish else patterns.append("HANGING MAN 🧗")

            if upper_wick > (body_size * 2) and lower_wick < (body_size * 0.5):
                patterns.append("SHOOTING STAR 🌠") if is_bearish else patterns.append("INV. HAMMER 🔨")

            if body_size <= (abs(c['high'] - c['low']) * 0.1): patterns.append("DOJI ➕")

            if is_bullish and prev['close'] < prev['open']: 
                 if open_p < prev['close'] and close_p > prev['open']: patterns.append("BULL ENGULFING 🦖")

            if is_bearish and prev['close'] > prev['open']:
                 if open_p > prev['close'] and close_p < prev['open']: patterns.append("BEAR ENGULFING 🐻")
        except: pass
                 
        return patterns

class NewsEngine:
    @staticmethod
    def generate_news(phase, price_change_pct, whale_ratio):
        news_templates = []
        if whale_ratio > 0.6:
            news_templates.extend([
                "🐋 BREAKING: Transaksi besar terdeteksi di Jaringan Haqq! Whale sedang mengakumulasi ISLM.",
                "💎 ON-CHAIN ALERT: Dompet 'Smart Money' baru saja menambah posisi ISLM."
            ])
        elif whale_ratio < 0.4:
            news_templates.extend([
                "⚠️ ALERT: Tekanan jual dari dompet paus terdeteksi. Waspada koreksi.",
                "📉 INFO PASAR: Beberapa validator besar terpantau melakukan aksi ambil untung (Profit Taking)."
            ])
        if phase == "MARKUP (NAIK) 🚀":
            news_templates.extend([
                "🚀 ADOPTION NEWS: Volume transaksi di ekosistem Islamic Coin meningkat 15% sejam terakhir.",
                "🌍 GLOBAL: Sentimen pasar aset syariah sedang positif di Timur Tengah."
            ])
        elif phase == "MARKDOWN (TURUN) 🔻":
            news_templates.extend([
                "🐻 SENTIMEN: Ketidakpastian global menekan harga aset kripto termasuk ISLM.",
                "🛑 SUPPORT TEST: Harga sedang menguji level kritikal. Hati-hati jebol."
            ])
        news_templates.extend([
            "ℹ️ TIPS: Diversifikasi aset tetap kunci aman trading.",
            "🕌 HAQQ UPDATE: Jaringan berjalan lancar. Block time stabil.",
            "📊 STATISTIK: Rasio Long/Short di exchange besar seimbang."
        ])
        try: return random.choice(news_templates)
        except: return "ℹ️ Info Pasar: Stay Safe."

class TelegramNotifier:
    def __init__(self, token, chat_id=None):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, message):
        if not self.chat_id: return False
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}
            threading.Thread(target=requests.post, args=(url,), kwargs={"json": payload}, daemon=True).start()
            return True
        except: return False

    def get_updates(self, offset=None):
        try:
            url = f"{self.base_url}/getUpdates"
            params = {"timeout": 10}
            if offset: params["offset"] = offset
            resp = requests.get(url, params=params, timeout=15).json()
            if resp.get("ok"):
                return resp.get("result", [])
        except: pass
        return []

class FundamentalEngine:
    @staticmethod
    def analyze_market_sentiment():
        # Simulated Islamic Finance News & Sentiment Analysis
        events = [
            ("Fatwa Dewan Syariah Nasional Positif", 5),
            ("Adopsi Islamic Coin di UAE Meningkat", 4),
            ("Kemitraan Baru dengan Bank Syariah", 3),
            ("Volume Transaksi Haqq Network Stabil", 1),
            ("Isu Regulasi Crypto Global (Netral)", 0),
            ("Koreksi Pasar Aset Digital Umum", -2),
            ("Volatilitas Harga Bitcoin Tinggi", -3),
            ("🌙 SENTIMEN RAMADHAN: Akumulasi Menjelang Bulan Suci (Bullish)", 8),
            ("Zakat & Infaq via Crypto Meningkat", 5)
        ]
        # Randomly pick 3 events (High probability of Ramadan appearing)
        daily_events = random.sample(events, 3)
        score = sum([e[1] for e in daily_events])
        news_text = " & ".join([e[0] for e in daily_events])
        return score, news_text

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
