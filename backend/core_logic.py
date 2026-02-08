import numpy as np
import random
import requests
import threading
import time
from datetime import datetime
from backend.config import Config

# --- Seed untuk reproduksi (opsional, bisa di-comment untuk true random)
# np.random.seed(42)

class MarketProjector:
    """Proyeksi harga berbasis Geometric Brownian Motion (GBM) + interval kepercayaan."""

    @staticmethod
    def calculate_volatility(prices):
        if len(prices) < 2:
            return 0.001
        pct_changes = np.diff(prices) / (prices[:-1] + 1e-10)
        return np.std(pct_changes)  # Std dev return = volatility

    @staticmethod
    def calculate_long_term_trend(prices):
        if len(prices) < 20:
            return 0
        x = np.arange(len(prices))
        y = np.array(prices, dtype=float)
        z = np.polyfit(x, y, 1)
        return z[0]

    @staticmethod
    def calculate_rsi_divergence(prices, rsi_values, window=10):
        if len(prices) < window or len(rsi_values) < window:
            return None
        price_slice = prices[-window:]
        rsi_slice = rsi_values[-window:]
        if price_slice[-1] < min(price_slice[:-1]) and rsi_slice[-1] > min(rsi_slice[:-1]):
            return "BULL_DIV"
        if price_slice[-1] > max(price_slice[:-1]) and rsi_slice[-1] < max(rsi_slice[:-1]):
            return "BEAR_DIV"
        return None

    @staticmethod
    def calculate_drift(prices):
        if len(prices) < 2:
            return 0
        pct_changes = np.diff(prices) / (prices[:-1] + 1e-10)
        return np.mean(pct_changes)

    @staticmethod
    def run_monte_carlo(current_price, volatility, drift, horizon_minutes, simulations=1000, steps=None):
        """Monte Carlo GBM. Returns (paths, percentiles_dict). steps=None => auto from horizon."""
        vol = max(volatility, 0.0005)
        if steps is None:
            steps = min(200, max(50, int(horizon_minutes / 30)))
        dt = horizon_minutes / (steps * 1440.0) if horizon_minutes > 0 else 1/1440.0  # in days
        step_drift = drift * dt
        step_vol = vol * np.sqrt(dt)
        shocks = np.random.normal(0, 1, (simulations, steps))
        log_returns = step_drift - 0.5 * step_vol**2 + step_vol * shocks
        price_paths = current_price * np.exp(np.cumsum(log_returns, axis=1))
        start_col = np.full((simulations, 1), current_price)
        price_paths = np.hstack((start_col, price_paths))
        final_prices = price_paths[:, -1]
        percentiles = {
            "p5": float(np.percentile(final_prices, 5)),
            "p25": float(np.percentile(final_prices, 25)),
            "p50": float(np.percentile(final_prices, 50)),
            "p75": float(np.percentile(final_prices, 75)),
            "p95": float(np.percentile(final_prices, 95)),
            "mean": float(np.mean(final_prices)),
        }
        return price_paths, percentiles

class CandleSniper:
    """Deteksi pola candlestick untuk sinyal reversal/continuation."""

    @staticmethod
    def _body(c): return abs(c['close'] - c['open'])
    @staticmethod
    def _range(c): return max(c['high'] - c['low'], 1e-10)
    @staticmethod
    def _upper_wick(c): return c['high'] - max(c['open'], c['close'])
    @staticmethod
    def _lower_wick(c): return min(c['open'], c['close']) - c['low']

    @staticmethod
    def analyze_patterns(candles):
        if len(candles) < 3:
            return []
        patterns = []
        try:
            c = candles[-1]
            prev = candles[-2]
            prev2 = candles[-3] if len(candles) >= 3 else None

            open_p, close_p = c['open'], c['close']
            high_p, low_p = c['high'], c['low']
            body_size = CandleSniper._body(c)
            upper_wick = CandleSniper._upper_wick(c)
            lower_wick = CandleSniper._lower_wick(c)
            rng = CandleSniper._range(c)
            is_bullish = close_p > open_p
            is_bearish = close_p < open_p

            # Hammer / Hanging Man
            if body_size <= rng * 0.35 and lower_wick >= body_size * 2 and upper_wick <= body_size * 0.5:
                patterns.append("HAMMER 🔨" if is_bullish else "HANGING MAN 🧗")
            # Shooting Star / Inverted Hammer
            if body_size <= rng * 0.35 and upper_wick >= body_size * 2 and lower_wick <= body_size * 0.5:
                patterns.append("SHOOTING STAR 🌠" if is_bearish else "INV. HAMMER 🔨")
            # Doji
            if body_size <= rng * 0.1:
                patterns.append("DOJI ➕")
            # Engulfing
            if is_bullish and prev['close'] < prev['open']:
                if open_p <= prev['close'] and close_p >= prev['open']:
                    patterns.append("BULL ENGULFING 🦖")
            if is_bearish and prev['close'] > prev['open']:
                if open_p >= prev['close'] and close_p <= prev['open']:
                    patterns.append("BEAR ENGULFING 🐻")
            # Morning Star (3-candle bullish)
            if prev2 and body_size > 0 and CandleSniper._body(prev) <= CandleSniper._range(prev) * 0.2:
                if prev2['close'] < prev2['open'] and close_p > (prev2['open'] + prev2['close']) / 2:
                    patterns.append("MORNING STAR 🌅")
            # Evening Star (3-candle bearish)
            if prev2 and body_size > 0 and CandleSniper._body(prev) <= CandleSniper._range(prev) * 0.2:
                if prev2['close'] > prev2['open'] and close_p < (prev2['open'] + prev2['close']) / 2:
                    patterns.append("EVENING STAR 🌆")
        except Exception:
            pass
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

class WhaleTracker:
    """Rasio order besar (whale) dari order book untuk deteksi pump/dump."""

    @staticmethod
    def get_whale_ratio(buy_orders, sell_orders, threshold_pct=0.1):
        """
        buy_orders, sell_orders: list of [price, amount].
        threshold_pct: consider top 10% by size as 'whale'.
        Returns whale_buy_ratio in [0,1]: >0.6 = whale buying, <0.4 = whale selling.
        """
        if not buy_orders or not sell_orders:
            return 0.5
        buy_vol = sum(float(a) for _, a in buy_orders)
        sell_vol = sum(float(a) for _, a in sell_orders)
        total = buy_vol + sell_vol
        if total == 0:
            return 0.5
        # Whale = top portion by size (simplified: sum of top 20% orders by value)
        def top_volume(orders, pct=0.2):
            by_val = sorted([(float(p) * float(a), a) for p, a in orders], key=lambda x: -x[0])
            n = max(1, int(len(by_val) * pct))
            return sum(a for _, a in by_val[:n])
        whale_buy = top_volume(buy_orders, threshold_pct)
        whale_sell = top_volume(sell_orders, threshold_pct)
        w_total = whale_buy + whale_sell
        if w_total == 0:
            return 0.5
        return whale_buy / w_total

    @staticmethod
    def interpret(ratio):
        if ratio >= 0.65:
            return "🐋 Whale Akumulasi (Buy Pressure)"
        if ratio <= 0.35:
            return "🐋 Whale Jual / Profit Taking"
        return "⚖️ Whale Seimbang"


class FundamentalEngine:
    """Sentimen pasar syariah + efek kalender (Ramadan)."""

    @staticmethod
    def _is_ramadan_zone():
        try:
            now = datetime.utcnow()
            # Approximate Ramadan 2025: ~Feb 28 - Mar 30 (converter bisa dipakai untuk akurat)
            # Simplifikasi: bulan 3 = Maret sering Ramadan
            if now.month == 3 or (now.month == 2 and now.day >= 25):
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def analyze_market_sentiment():
        events = [
            ("Fatwa Dewan Syariah Nasional Positif", 5),
            ("Adopsi Islamic Coin di UAE Meningkat", 4),
            ("Kemitraan Baru dengan Bank Syariah", 3),
            ("Volume Transaksi Haqq Network Stabil", 1),
            ("Isu Regulasi Crypto Global (Netral)", 0),
            ("Koreksi Pasar Aset Digital Umum", -2),
            ("Volatilitas Harga Bitcoin Tinggi", -3),
            ("🌙 SENTIMEN RAMADHAN: Akumulasi Menjelang Bulan Suci (Bullish)", 8),
            ("Zakat & Infaq via Crypto Meningkat", 5),
        ]
        daily_events = random.sample(events, min(3, len(events)))
        score = sum(e[1] for e in daily_events)
        if FundamentalEngine._is_ramadan_zone():
            score = min(10, score + 2)
        news_text = " & ".join(e[0] for e in daily_events)
        return score, news_text

class QuantAnalyzer:
    """Indikator teknikal: RSI (Wilder), MACD (benar), Bollinger, ATR, Fibonacci, fase pasar."""

    @staticmethod
    def calculate_ema_series(prices, window):
        """Return array EMA untuk tiap titik (untuk MACD signal)."""
        if len(prices) < window:
            return None
        out = np.zeros(len(prices))
        mult = 2.0 / (window + 1)
        out[window - 1] = np.mean(prices[:window])
        for i in range(window, len(prices)):
            out[i] = (prices[i] - out[i - 1]) * mult + out[i - 1]
        return out

    @staticmethod
    def calculate_rsi(prices, period=14):
        """RSI dengan smoothing ala Wilder (EMA gains/losses)."""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def calculate_rsi_series(prices, period=14):
        """Return array RSI untuk divergence check."""
        if len(prices) < period + 1:
            return None
        out = np.full(len(prices), 50.0)
        deltas = np.diff(prices)
        for i in range(period, len(deltas)):
            g = deltas[i - period + 1 : i + 1]
            gains = np.where(g > 0, g, 0)
            losses = np.where(g < 0, -g, 0)
            ag, al = np.mean(gains), np.mean(losses)
            if al == 0:
                out[i + 1] = 100.0
            else:
                out[i + 1] = 100.0 - 100.0 / (1.0 + ag / al)
        return out

    @staticmethod
    def calculate_stoch_rsi(prices, period=14):
        if len(prices) < period:
            return 50.0, 50.0
        recent = prices[-period:]
        h, l = np.max(recent), np.min(recent)
        if h == l:
            return 50.0, 50.0
        stoch = ((prices[-1] - l) / (h - l)) * 100
        return stoch, stoch

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal_period=9):
        """MACD line, Signal = EMA(signal_period) dari MACD line, Histogram."""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        ema_f = QuantAnalyzer.calculate_ema_series(prices, fast)
        ema_s = QuantAnalyzer.calculate_ema_series(prices, slow)
        macd_line = ema_f - ema_s
        # Signal = EMA of MACD
        macd_series = macd_line[slow - 1 :]
        if len(macd_series) < signal_period:
            return float(macd_line[-1]), float(macd_line[-1]), 0.0
        signal_ema = QuantAnalyzer.calculate_ema_series(macd_series, signal_period)
        signal_line = signal_ema[-1] if signal_ema is not None else macd_line[-1]
        macd_val = macd_line[-1]
        hist = macd_val - signal_line
        return float(macd_val), float(signal_line), float(hist)

    @staticmethod
    def calculate_fibonacci(high, low):
        diff = high - low
        return {
            "0": high,
            "0.236": high - diff * 0.236,
            "0.382": high - diff * 0.382,
            "0.5": high - diff * 0.5,
            "0.618": high - diff * 0.618,
            "0.786": high - diff * 0.786,
            "1": low,
        }

    @staticmethod
    def calculate_sma(prices, window):
        if len(prices) < window:
            return None
        return float(np.mean(prices[-window:]))

    @staticmethod
    def calculate_ema(prices, window):
        if len(prices) < window:
            return None
        weights = np.exp(np.linspace(-1.0, 0.0, window))
        weights /= weights.sum()
        a = np.convolve(prices[-window:], weights, mode="full")[: len(prices[-window:])]
        return float(a[-1])

    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        if len(prices) < window:
            return None, None, None
        sma = np.mean(prices[-window:])
        std = np.std(prices[-window:])
        return sma + std * num_std, sma, sma - std * num_std

    @staticmethod
    def calculate_atr(highs, lows, closes, period=14):
        if len(highs) < period + 1:
            return None
        tr = []
        for i in range(1, len(highs)):
            tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        return float(np.mean(tr[-period:]))

    @staticmethod
    def detect_market_phase(prices, vol_proxy=None):
        if len(prices) < 20:
            return "NETRAL"
        sma20 = np.mean(prices[-20:])
        current = prices[-1]
        pct = abs(current - sma20) / (sma20 + 1e-10)
        if pct < 0.01:
            return "AKUMULASI 🧱"
        if current > sma20 * 1.02:
            return "MARKUP (NAIK) 🚀"
        if current < sma20 * 0.98:
            return "MARKDOWN (TURUN) 🔻"
        return "KONSOLIDASI ⚖️"

    @staticmethod
    def get_psychological_level(price):
        levels = [150, 200, 225, 250, 275, 300, 350, 400, 500, 750, 1000]
        closest = min(levels, key=lambda x: abs(x - price))
        if abs(price - closest) / (price + 1e-10) < 0.02:
            return closest
        return None


class AISignalEngine:
    """Gabungan RSI, MACD, BB, Candle, Whale, Fundamental -> satu skor & label sinyal AI."""

    @staticmethod
    def compute(
        rsi,
        macd_hist,
        price,
        bb_mid,
        bb_upper,
        bb_lower,
        candle_bull_count,
        candle_bear_count,
        whale_ratio,
        fundamental_score,
    ):
        score = 0.0  # -1 s/d 1
        if rsi < 30:
            score += 0.25
        elif rsi > 70:
            score -= 0.25
        elif rsi < 45:
            score += 0.08
        elif rsi > 55:
            score -= 0.08
        if macd_hist > 0:
            score += 0.15
        else:
            score -= 0.15
        if bb_mid is not None:
            if price < (bb_lower or 0):
                score += 0.15
            elif price > (bb_upper or 0):
                score -= 0.15
        score += (candle_bull_count - candle_bear_count) * 0.1
        if whale_ratio >= 0.6:
            score += 0.12
        elif whale_ratio <= 0.4:
            score -= 0.12
        if fundamental_score >= 6:
            score += 0.1
        elif fundamental_score <= 2:
            score -= 0.1
        score = max(-1.0, min(1.0, score))
        if score >= 0.4:
            label = "STRONG BUY 🚀"
        elif score >= 0.15:
            label = "BUY 📈"
        elif score <= -0.4:
            label = "STRONG SELL ⚠️"
        elif score <= -0.15:
            label = "SELL 📉"
        else:
            label = "HOLD 🤝"
        return {"score": score, "label": label, "confidence": abs(score)}
