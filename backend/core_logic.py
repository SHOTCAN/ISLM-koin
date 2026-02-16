"""
ISLM Monitor — AI Analytics Engine V4
=====================================
Enhanced with:
  - scikit-learn: ML-based signal classification
  - ta: Professional technical analysis (40+ indicators)
  - Multi-horizon predictions (GBM Monte Carlo)
  - Support/Resistance detection
  - Volume-Weighted analysis
  - Explainable reasoning
"""

import numpy as np
import pandas as pd
import random
import requests
import threading
import time
from datetime import datetime

try:
    import ta as ta_lib
    HAS_TA = True
except ImportError:
    HAS_TA = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ============================================
# MARKET PROJECTOR (Monte Carlo GBM)
# ============================================
class MarketProjector:
    """Proyeksi harga berbasis Geometric Brownian Motion (GBM)."""

    @staticmethod
    def calculate_volatility(prices):
        if len(prices) < 2: return 0.001
        pct = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
        return float(np.std(pct))

    @staticmethod
    def calculate_drift(prices):
        if len(prices) < 2: return 0
        pct = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
        return float(np.mean(pct))

    @staticmethod
    def calculate_long_term_trend(prices):
        if len(prices) < 20: return 0
        x = np.arange(len(prices))
        z = np.polyfit(x, np.array(prices, dtype=float), 1)
        return z[0]

    @staticmethod
    def run_monte_carlo(current_price, volatility, drift, horizon_minutes, simulations=1000):
        vol = max(volatility, 0.0005)
        steps = min(200, max(50, int(horizon_minutes / 30)))
        dt = horizon_minutes / (steps * 1440.0) if horizon_minutes > 0 else 1 / 1440.0
        shocks = np.random.normal(0, 1, (simulations, steps))
        log_ret = (drift * dt - 0.5 * vol ** 2 * dt) + vol * np.sqrt(dt) * shocks
        paths = current_price * np.exp(np.cumsum(log_ret, axis=1))
        start = np.full((simulations, 1), current_price)
        paths = np.hstack((start, paths))
        finals = paths[:, -1]
        return paths, {
            "p5": float(np.percentile(finals, 5)),
            "p25": float(np.percentile(finals, 25)),
            "p50": float(np.percentile(finals, 50)),
            "p75": float(np.percentile(finals, 75)),
            "p95": float(np.percentile(finals, 95)),
            "mean": float(np.mean(finals)),
        }

    @staticmethod
    def predict_multi_horizon(current_price, prices):
        vol = MarketProjector.calculate_volatility(np.array(prices))
        drift = MarketProjector.calculate_drift(np.array(prices))
        trend = MarketProjector.calculate_long_term_trend(np.array(prices))
        adj_drift = drift + (trend / (current_price + 1e-10)) * 0.1
        horizons = {
            "1_hari": {"minutes": 1440, "sims": 1000, "label": "1 Hari"},
            "3_hari": {"minutes": 4320, "sims": 1000, "label": "3 Hari"},
            "7_hari": {"minutes": 10080, "sims": 800, "label": "7 Hari"},
        }
        results = {}
        for key, h in horizons.items():
            _, pct = MarketProjector.run_monte_carlo(current_price, vol, adj_drift, h["minutes"], h["sims"])
            chg = (pct["p50"] - current_price) / (current_price + 1e-10) * 100
            if chg > 2: direction = "NAIK 📈"
            elif chg < -2: direction = "TURUN 📉"
            else: direction = "SIDEWAYS ➡️"
            results[key] = {
                "label": h["label"], "target": pct["p50"], "low": pct["p5"], "high": pct["p95"],
                "change_pct": chg, "direction": direction,
                "confidence": min(95, max(40, 80 - abs(chg) * 2)),
            }
        return results


# ============================================
# PROFESSIONAL TECHNICAL ANALYSIS (ta library)
# ============================================
class ProTA:
    """Professional TA using the 'ta' library — 40+ indicators."""

    @staticmethod
    def compute_all(df):
        """Compute all available TA indicators on a DataFrame with OHLCV columns."""
        if not HAS_TA or df is None or len(df) < 30:
            return {}
        try:
            result = {}
            high = df['high']
            low = df['low']
            close = df['close']
            vol = df['vol'] if 'vol' in df.columns else pd.Series(np.zeros(len(df)))

            # --- Trend Indicators ---
            result['ema_9'] = float(ta_lib.trend.ema_indicator(close, window=9).iloc[-1])
            result['ema_21'] = float(ta_lib.trend.ema_indicator(close, window=21).iloc[-1])
            result['sma_20'] = float(ta_lib.trend.sma_indicator(close, window=20).iloc[-1])
            result['sma_50'] = float(ta_lib.trend.sma_indicator(close, window=50).iloc[-1]) if len(df) >= 50 else None

            # ADX (Average Directional Index) — trend strength
            adx_ind = ta_lib.trend.ADXIndicator(high, low, close, window=14)
            result['adx'] = float(adx_ind.adx().iloc[-1])
            result['adx_pos'] = float(adx_ind.adx_pos().iloc[-1])
            result['adx_neg'] = float(adx_ind.adx_neg().iloc[-1])

            # Ichimoku
            try:
                ich = ta_lib.trend.IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
                result['ichimoku_a'] = float(ich.ichimoku_a().iloc[-1])
                result['ichimoku_b'] = float(ich.ichimoku_b().iloc[-1])
                result['ichimoku_base'] = float(ich.ichimoku_base_line().iloc[-1])
            except:
                pass

            # --- Momentum Indicators ---
            result['rsi'] = float(ta_lib.momentum.rsi(close, window=14).iloc[-1])
            result['stoch_k'] = float(ta_lib.momentum.stoch(high, low, close).iloc[-1])
            result['stoch_d'] = float(ta_lib.momentum.stoch_signal(high, low, close).iloc[-1])

            # Williams %R
            result['williams_r'] = float(ta_lib.momentum.williams_r(high, low, close).iloc[-1])

            # ROC (Rate of Change)
            result['roc'] = float(ta_lib.momentum.roc(close, window=12).iloc[-1])

            # --- Volatility Indicators ---
            bb = ta_lib.volatility.BollingerBands(close, window=20)
            result['bb_upper'] = float(bb.bollinger_hband().iloc[-1])
            result['bb_mid'] = float(bb.bollinger_mavg().iloc[-1])
            result['bb_lower'] = float(bb.bollinger_lband().iloc[-1])
            result['bb_width'] = float(bb.bollinger_wband().iloc[-1])
            result['bb_pct'] = float(bb.bollinger_pband().iloc[-1])

            # ATR
            result['atr'] = float(ta_lib.volatility.average_true_range(high, low, close).iloc[-1])

            # --- Volume Indicators ---
            if vol.sum() > 0:
                result['obv'] = float(ta_lib.volume.on_balance_volume(close, vol).iloc[-1])
                result['vwap'] = float(ta_lib.volume.volume_weighted_average_price(high, low, close, vol).iloc[-1])
                result['mfi'] = float(ta_lib.volume.money_flow_index(high, low, close, vol).iloc[-1])
                result['fi'] = float(ta_lib.volume.force_index(close, vol).iloc[-1])

            # --- MACD ---
            macd_ind = ta_lib.trend.MACD(close)
            result['macd'] = float(macd_ind.macd().iloc[-1])
            result['macd_signal'] = float(macd_ind.macd_signal().iloc[-1])
            result['macd_hist'] = float(macd_ind.macd_diff().iloc[-1])

            return result
        except Exception as e:
            print(f"[ProTA Error] {e}")
            return {}


# ============================================
# ML SIGNAL CLASSIFIER (scikit-learn)
# ============================================
class MLSignalClassifier:
    """ML-based signal classification using GradientBoosting."""

    @staticmethod
    def build_features(df):
        """Build feature matrix from OHLCV DataFrame."""
        if df is None or len(df) < 50:
            return None, None

        try:
            close = df['close'].values
            features = []
            labels = []

            for i in range(30, len(close) - 5):
                window = close[i - 30:i]
                future = close[i:i + 5]

                pct_changes = np.diff(window) / (window[:-1] + 1e-10)

                # Features
                rsi = MLSignalClassifier._rsi(window)
                sma_ratio = window[-1] / (np.mean(window) + 1e-10)
                vol = np.std(pct_changes)
                momentum = (window[-1] - window[-5]) / (window[-5] + 1e-10) if len(window) >= 5 else 0
                trend = np.polyfit(np.arange(len(window)), window, 1)[0]

                feat = [rsi, sma_ratio, vol, momentum, trend,
                        np.mean(pct_changes), np.max(pct_changes), np.min(pct_changes)]
                features.append(feat)

                # Label: 1=BUY, 0=HOLD, -1=SELL
                future_change = (np.mean(future) - window[-1]) / (window[-1] + 1e-10)
                if future_change > 0.01:
                    labels.append(1)
                elif future_change < -0.01:
                    labels.append(-1)
                else:
                    labels.append(0)

            return np.array(features), np.array(labels)
        except:
            return None, None

    @staticmethod
    def _rsi(prices, period=14):
        if len(prices) < period + 1: return 50.0
        d = np.diff(prices)
        g, l = np.where(d > 0, d, 0), np.where(d < 0, -d, 0)
        ag, al = np.mean(g[-period:]), np.mean(l[-period:])
        if al == 0: return 100.0
        return 100.0 - 100.0 / (1.0 + ag / al)

    @staticmethod
    def predict_signal(df):
        """Train on historical data and predict current signal."""
        if not HAS_SKLEARN or df is None or len(df) < 60:
            return {"ml_signal": "N/A", "ml_confidence": 0, "ml_available": False}

        try:
            X, y = MLSignalClassifier.build_features(df)
            if X is None or len(X) < 20:
                return {"ml_signal": "N/A", "ml_confidence": 0, "ml_available": False}

            # Normalize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train on all but last point
            X_train, y_train = X_scaled[:-1], y[:-1]
            X_pred = X_scaled[-1:].reshape(1, -1)

            # Train GBM
            model = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
            )
            model.fit(X_train, y_train)

            # Predict
            pred = model.predict(X_pred)[0]
            proba = model.predict_proba(X_pred)[0]
            confidence = float(np.max(proba))

            signal_map = {1: "BUY 📈", 0: "HOLD 🤝", -1: "SELL 📉"}
            return {
                "ml_signal": signal_map.get(pred, "HOLD 🤝"),
                "ml_confidence": confidence,
                "ml_available": True,
                "ml_class_probs": {signal_map.get(c, "?"): float(p) for c, p in zip(model.classes_, proba)},
            }
        except Exception as e:
            print(f"[ML Error] {e}")
            return {"ml_signal": "N/A", "ml_confidence": 0, "ml_available": False}


# ============================================
# SUPPORT & RESISTANCE DETECTOR
# ============================================
class SupportResistance:
    """Detect key support and resistance levels."""

    @staticmethod
    def find_levels(df, window=20):
        if df is None or len(df) < window:
            return [], []
        highs = df['high'].values
        lows = df['low'].values
        supports = []
        resistances = []

        for i in range(window, len(highs) - window):
            # Resistance: local max
            if highs[i] == max(highs[i - window:i + window + 1]):
                resistances.append(float(highs[i]))
            # Support: local min
            if lows[i] == min(lows[i - window:i + window + 1]):
                supports.append(float(lows[i]))

        # Cluster nearby levels
        supports = SupportResistance._cluster(supports)
        resistances = SupportResistance._cluster(resistances)
        return supports[-3:], resistances[-3:]  # Top 3

    @staticmethod
    def _cluster(levels, threshold=0.02):
        if not levels: return []
        levels.sort()
        clustered = [levels[0]]
        for l in levels[1:]:
            if abs(l - clustered[-1]) / (clustered[-1] + 1e-10) > threshold:
                clustered.append(l)
            else:
                clustered[-1] = (clustered[-1] + l) / 2
        return clustered


# ============================================
# CANDLE PATTERN SNIPER
# ============================================
class CandleSniper:
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
        if len(candles) < 3: return []
        patterns = []
        try:
            c, prev = candles[-1], candles[-2]
            prev2 = candles[-3] if len(candles) >= 3 else None
            body = CandleSniper._body(c)
            rng = CandleSniper._range(c)
            uwick = CandleSniper._upper_wick(c)
            lwick = CandleSniper._lower_wick(c)
            bullish = c['close'] > c['open']
            bearish = c['close'] < c['open']
            if body <= rng * 0.35 and lwick >= body * 2 and uwick <= body * 0.5:
                patterns.append("HAMMER 🔨" if bullish else "HANGING MAN 🧗")
            if body <= rng * 0.35 and uwick >= body * 2 and lwick <= body * 0.5:
                patterns.append("SHOOTING STAR 🌠" if bearish else "INV. HAMMER 🔨")
            if body <= rng * 0.1:
                patterns.append("DOJI ➕")
            if bullish and prev['close'] < prev['open'] and c['open'] <= prev['close'] and c['close'] >= prev['open']:
                patterns.append("BULL ENGULFING 🦖")
            if bearish and prev['close'] > prev['open'] and c['open'] >= prev['close'] and c['close'] <= prev['open']:
                patterns.append("BEAR ENGULFING 🐻")
            if prev2 and body > 0 and CandleSniper._body(prev) <= CandleSniper._range(prev) * 0.2:
                if prev2['close'] < prev2['open'] and c['close'] > (prev2['open'] + prev2['close']) / 2:
                    patterns.append("MORNING STAR 🌅")
                if prev2['close'] > prev2['open'] and c['close'] < (prev2['open'] + prev2['close']) / 2:
                    patterns.append("EVENING STAR 🌆")
        except: pass
        return patterns


# ============================================
# NEWS ENGINE
# ============================================
class NewsEngine:
    @staticmethod
    def generate_news(phase, whale_ratio):
        templates = []
        if whale_ratio > 0.6:
            templates += ["🐋 BREAKING: Whale mengakumulasi ISLM.", "💎 Smart Money menambah posisi ISLM."]
        elif whale_ratio < 0.4:
            templates += ["⚠️ Tekanan jual whale terdeteksi.", "📉 Validator besar ambil untung."]
        if "NAIK" in phase:
            templates += ["🚀 Volume transaksi ekosistem Islamic Coin meningkat.", "🌍 Sentimen aset syariah positif."]
        elif "TURUN" in phase:
            templates += ["🐻 Ketidakpastian global menekan ISLM.", "🛑 Harga menguji support."]
        templates += ["ℹ️ Diversifikasi tetap kunci aman.", "🕌 Haqq Network berjalan stabil.", "📊 Rasio Long/Short seimbang."]
        return random.choice(templates)


# ============================================
# WHALE TRACKER
# ============================================
class WhaleTracker:
    @staticmethod
    def get_whale_ratio(buy_orders, sell_orders, threshold_pct=0.1):
        if not buy_orders or not sell_orders: return 0.5
        buy_vol = sum(float(a) for _, a in buy_orders)
        sell_vol = sum(float(a) for _, a in sell_orders)
        total = buy_vol + sell_vol
        if total == 0: return 0.5
        def top_vol(orders, pct=0.2):
            by_val = sorted([(float(p) * float(a), float(a)) for p, a in orders], key=lambda x: -x[0])
            n = max(1, int(len(by_val) * pct))
            return sum(a for _, a in by_val[:n])
        wb, ws = top_vol(buy_orders, threshold_pct), top_vol(sell_orders, threshold_pct)
        wt = wb + ws
        return wb / wt if wt > 0 else 0.5

    @staticmethod
    def interpret(ratio):
        if ratio >= 0.65: return "🐋 Whale Akumulasi (Buy Pressure)"
        if ratio <= 0.35: return "🐋 Whale Jual / Profit Taking"
        return "⚖️ Whale Seimbang"


# ============================================
# FUNDAMENTAL ENGINE
# ============================================
class FundamentalEngine:
    @staticmethod
    def analyze_market_sentiment():
        events = [
            ("Fatwa Dewan Syariah Nasional Positif", 5),
            ("Adopsi Islamic Coin di UAE Meningkat", 4),
            ("Kemitraan Baru dengan Bank Syariah", 3),
            ("Volume Transaksi Haqq Network Stabil", 1),
            ("Isu Regulasi Crypto Global (Netral)", 0),
            ("Koreksi Pasar Aset Digital Umum", -2),
            ("Volatilitas Bitcoin Tinggi", -3),
            ("🌙 Akumulasi Menjelang Bulan Suci (Bullish)", 8),
            ("Zakat & Infaq via Crypto Meningkat", 5),
        ]
        daily = random.sample(events, min(3, len(events)))
        score = sum(e[1] for e in daily)
        return min(10, max(0, score)), " & ".join(e[0] for e in daily)


# ============================================
# QUANT ANALYZER (Manual TA - Fallback)
# ============================================
class QuantAnalyzer:
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) < period + 1: return 50.0
        d = np.diff(prices)
        g, l = np.where(d > 0, d, 0), np.where(d < 0, -d, 0)
        return 100.0 - 100.0 / (1 + np.mean(g[-period:]) / (np.mean(l[-period:]) + 1e-10))

    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, sig=9):
        if len(prices) < slow: return 0.0, 0.0, 0.0
        def ema(p, w):
            out = np.zeros(len(p)); m = 2 / (w + 1)
            out[w - 1] = np.mean(p[:w])
            for i in range(w, len(p)): out[i] = (p[i] - out[i - 1]) * m + out[i - 1]
            return out
        ef, es = ema(prices, fast), ema(prices, slow)
        ml = ef - es
        ms = ml[slow - 1:]
        if len(ms) < sig: return float(ml[-1]), float(ml[-1]), 0.0
        se = ema(ms, sig)
        return float(ml[-1]), float(se[-1] if se is not None else ml[-1]), float(ml[-1] - (se[-1] if se is not None else ml[-1]))

    @staticmethod
    def calculate_bollinger_bands(prices, w=20, n=2):
        if len(prices) < w: return None, None, None
        s, st = np.mean(prices[-w:]), np.std(prices[-w:])
        return s + st * n, s, s - st * n

    @staticmethod
    def calculate_atr(highs, lows, closes, period=14):
        if len(highs) < period + 1: return None
        tr = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])) for i in range(1, len(highs))]
        return float(np.mean(tr[-period:]))

    @staticmethod
    def calculate_stoch_rsi(prices, period=14):
        if len(prices) < period: return 50.0, 50.0
        r = prices[-period:]
        h, l = np.max(r), np.min(r)
        if h == l: return 50.0, 50.0
        return ((prices[-1] - l) / (h - l)) * 100, ((prices[-1] - l) / (h - l)) * 100

    @staticmethod
    def calculate_fibonacci(high, low):
        d = high - low
        return {"0": high, "0.236": high - d * 0.236, "0.382": high - d * 0.382,
                "0.5": high - d * 0.5, "0.618": high - d * 0.618, "0.786": high - d * 0.786, "1": low}

    @staticmethod
    def detect_market_phase(prices):
        if len(prices) < 20: return "NETRAL"
        sma20, cur = np.mean(prices[-20:]), prices[-1]
        pct = abs(cur - sma20) / (sma20 + 1e-10)
        if pct < 0.01: return "AKUMULASI 🧱"
        if cur > sma20 * 1.02: return "MARKUP (NAIK) 🚀"
        if cur < sma20 * 0.98: return "MARKDOWN (TURUN) 🔻"
        return "KONSOLIDASI ⚖️"


# ============================================
# BTC CORRELATION ANALYZER (V5)
# ============================================
class BTCCorrelation:
    """Analisis korelasi ISLM vs BTC untuk sinyal cross-market."""

    @staticmethod
    def fetch_btc_price():
        """Ambil harga BTC/IDR dari Indodax."""
        try:
            r = requests.get('https://indodax.com/api/ticker/btcidr', timeout=8)
            return float(r.json().get('ticker', {}).get('last', 0))
        except:
            return 0

    @staticmethod
    def calculate_correlation(prices_a, prices_b):
        """Hitung korelasi Pearson antara dua deret harga."""
        if len(prices_a) < 10 or len(prices_b) < 10:
            return 0.0
        n = min(len(prices_a), len(prices_b))
        a, b = np.array(prices_a[-n:], dtype=float), np.array(prices_b[-n:], dtype=float)
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    @staticmethod
    def interpret(corr):
        """Interpretasi korelasi."""
        if corr > 0.7: return f"KORELASI TINGGI ({corr:.2f}) — ISLM ikut BTC 📊"
        if corr > 0.3: return f"KORELASI SEDANG ({corr:.2f}) — ISLM agak ikut BTC 🔗"
        if corr > -0.3: return f"KORELASI RENDAH ({corr:.2f}) — ISLM independen 🆓"
        return f"KORELASI NEGATIF ({corr:.2f}) — ISLM berlawanan BTC ↕️"


# ============================================
# RISK METRICS (V5)
# ============================================
class RiskMetrics:
    """Metrik risiko: Sharpe, MaxDrawdown, WinRate, VaR."""

    @staticmethod
    def sharpe_ratio(prices, risk_free_rate=0.0):
        """Sharpe Ratio — return per unit risiko."""
        if len(prices) < 5: return 0.0
        returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
        excess = returns - risk_free_rate / 365
        std = np.std(excess)
        if std < 1e-10: return 0.0
        return float(np.mean(excess) / std * np.sqrt(365))

    @staticmethod
    def max_drawdown(prices):
        """Max Drawdown — kerugian max dari peak."""
        if len(prices) < 2: return 0.0
        peak = np.maximum.accumulate(prices)
        dd = (peak - prices) / (peak + 1e-10) * 100
        return float(np.max(dd))

    @staticmethod
    def win_rate(prices):
        """Win Rate — persentase candle hijau."""
        if len(prices) < 2: return 50.0
        returns = np.diff(prices)
        wins = np.sum(returns > 0)
        return float(wins / len(returns) * 100)

    @staticmethod
    def value_at_risk(prices, confidence=0.95):
        """VaR — potensi kerugian pada confidence level."""
        if len(prices) < 10: return 0.0
        returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-10)
        return float(np.percentile(returns, (1 - confidence) * 100) * 100)

    @staticmethod
    def full_report(prices):
        """Laporan risiko lengkap."""
        return {
            'sharpe': RiskMetrics.sharpe_ratio(prices),
            'max_dd': RiskMetrics.max_drawdown(prices),
            'win_rate': RiskMetrics.win_rate(prices),
            'var_95': RiskMetrics.value_at_risk(prices, 0.95),
        }


# ============================================
# TREND STRENGTH INDEX (V5)
# ============================================
class TrendStrengthIndex:
    """Gabungan ADX + EMA slope + volume trend = skor kekuatan trend."""

    @staticmethod
    def calculate(pro_ta, prices, volumes=None):
        """Hitung Trend Strength 0-100."""
        score = 50  # Netral

        # ADX contribution (0-30 points)
        adx = pro_ta.get('adx', 25) if pro_ta else 25
        if adx > 40: score += 25
        elif adx > 25: score += 15
        elif adx < 15: score -= 10

        # EMA slope (0-20 points)
        if len(prices) >= 21:
            ema9 = float(pd.Series(prices).ewm(span=9).mean().iloc[-1])
            ema21 = float(pd.Series(prices).ewm(span=21).mean().iloc[-1])
            if ema9 > ema21: score += 15
            else: score -= 10

        # Price momentum (0-15 points)
        if len(prices) >= 5:
            momentum = (prices[-1] - prices[-5]) / (prices[-5] + 1e-10) * 100
            if momentum > 3: score += 15
            elif momentum > 1: score += 8
            elif momentum < -3: score -= 15
            elif momentum < -1: score -= 8

        # Volume trend (0-10 points)
        if volumes and len(volumes) >= 10:
            recent_vol = np.mean(volumes[-5:])
            older_vol = np.mean(volumes[-10:-5])
            if recent_vol > older_vol * 1.3: score += 10
            elif recent_vol < older_vol * 0.7: score -= 5

        return max(0, min(100, score))

    @staticmethod
    def interpret(score):
        if score >= 80: return "TREND SANGAT KUAT 🔥🔥🔥"
        if score >= 60: return "TREND KUAT 🔥🔥"
        if score >= 40: return "TREND SEDANG 🔥"
        if score >= 20: return "TREND LEMAH 💤"
        return "TIDAK ADA TREND ❌"


# ============================================
# UNIFIED AI SIGNAL ENGINE V4
# ============================================
class AISignalEngine:
    """Gabungan Rule-Based + ML + ProTA = Sinyal AI V4 dengan Reasoning."""

    @staticmethod
    def compute(rsi, macd_hist, price, bb_mid, bb_upper, bb_lower,
                candle_bull_count, candle_bear_count, whale_ratio, fundamental_score,
                pro_ta=None, ml_result=None):
        score = 0.0
        reasons = []

        # --- RSI (25%) ---
        if rsi < 30:
            score += 0.25; reasons.append(f"📊 RSI={rsi:.0f} → Oversold (Jenuh Jual)")
        elif rsi > 70:
            score -= 0.25; reasons.append(f"📊 RSI={rsi:.0f} → Overbought (Jenuh Beli)")
        elif rsi < 45:
            score += 0.08; reasons.append(f"📊 RSI={rsi:.0f} → Cenderung Bullish")
        elif rsi > 55:
            score -= 0.08; reasons.append(f"📊 RSI={rsi:.0f} → Cenderung Bearish")
        else:
            reasons.append(f"📊 RSI={rsi:.0f} → Netral")

        # --- MACD (15%) ---
        if macd_hist > 0:
            score += 0.15; reasons.append(f"📈 MACD Histogram +{macd_hist:.2f} → Momentum Naik")
        else:
            score -= 0.15; reasons.append(f"📉 MACD Histogram {macd_hist:.2f} → Momentum Turun")

        # --- Bollinger Bands (15%) ---
        if bb_mid is not None and bb_lower is not None and bb_upper is not None:
            if price < bb_lower:
                score += 0.15; reasons.append("🔽 Harga di BAWAH BB Lower → Potensi Rebound")
            elif price > bb_upper:
                score -= 0.15; reasons.append("🔼 Harga di ATAS BB Upper → Potensi Koreksi")
            else:
                bb_pos = (price - bb_lower) / (bb_upper - bb_lower + 1e-10) * 100
                reasons.append(f"📐 Posisi BB: {bb_pos:.0f}%")

        # --- Candle Patterns (10%) ---
        net = candle_bull_count - candle_bear_count
        score += net * 0.1
        if candle_bull_count > 0: reasons.append(f"🕯️ {candle_bull_count} Pola Bullish")
        if candle_bear_count > 0: reasons.append(f"🕯️ {candle_bear_count} Pola Bearish")

        # --- Whale (12%) ---
        if whale_ratio >= 0.6:
            score += 0.12; reasons.append(f"🐋 Whale BELI dominan ({whale_ratio*100:.0f}%) → Akumulasi")
        elif whale_ratio <= 0.4:
            score -= 0.12; reasons.append(f"🐋 Whale JUAL dominan ({whale_ratio*100:.0f}%) → Distribusi")
        else:
            reasons.append(f"🐋 Whale seimbang ({whale_ratio*100:.0f}%)")

        # --- Fundamental (10%) ---
        if fundamental_score >= 6:
            score += 0.1; reasons.append(f"🌙 Fundamental KUAT (Skor: {fundamental_score}/10)")
        elif fundamental_score <= 2:
            score -= 0.1; reasons.append(f"⚠️ Fundamental LEMAH (Skor: {fundamental_score}/10)")
        else:
            reasons.append(f"📋 Fundamental Netral (Skor: {fundamental_score}/10)")

        # --- ProTA Extras (if available) ---
        if pro_ta:
            # ADX trend strength
            adx = pro_ta.get('adx', 0)
            if adx > 25:
                reasons.append(f"💪 Trend KUAT (ADX={adx:.0f})")
                if pro_ta.get('adx_pos', 0) > pro_ta.get('adx_neg', 0):
                    score += 0.05
                else:
                    score -= 0.05
            elif adx < 20:
                reasons.append(f"😶 Trend LEMAH (ADX={adx:.0f}) → Sideways")

            # EMA Cross
            ema9 = pro_ta.get('ema_9')
            ema21 = pro_ta.get('ema_21')
            if ema9 and ema21:
                if ema9 > ema21:
                    score += 0.05; reasons.append("✨ EMA9 > EMA21 → Golden Cross")
                else:
                    score -= 0.05; reasons.append("💀 EMA9 < EMA21 → Death Cross")

            # MFI (Money Flow Index)
            mfi = pro_ta.get('mfi')
            if mfi is not None:
                if mfi < 20:
                    score += 0.05; reasons.append(f"💰 MFI={mfi:.0f} → Uang Masuk (Oversold)")
                elif mfi > 80:
                    score -= 0.05; reasons.append(f"💸 MFI={mfi:.0f} → Uang Keluar (Overbought)")

            # Williams %R
            wr = pro_ta.get('williams_r')
            if wr is not None:
                if wr < -80:
                    reasons.append(f"📉 Williams %R={wr:.0f} → Oversold")
                elif wr > -20:
                    reasons.append(f"📈 Williams %R={wr:.0f} → Overbought")

        # --- ML Signal (if available) ---
        if ml_result and ml_result.get('ml_available'):
            ml_sig = ml_result['ml_signal']
            ml_conf = ml_result['ml_confidence']
            if "BUY" in ml_sig:
                score += 0.1 * ml_conf
            elif "SELL" in ml_sig:
                score -= 0.1 * ml_conf
            reasons.append(f"🤖 ML Model: {ml_sig} ({ml_conf*100:.0f}% confidence)")

        # --- Final ---
        score = max(-1.0, min(1.0, score))
        confidence = abs(score)
        if score >= 0.4: label = "STRONG BUY 🚀"
        elif score >= 0.15: label = "BUY 📈"
        elif score <= -0.4: label = "STRONG SELL ⚠️"
        elif score <= -0.15: label = "SELL 📉"
        else: label = "HOLD 🤝"

        trend = "Sideways ➡️"
        if bb_mid is not None:
            if price > bb_mid * 1.01: trend = "Bullish 🐂"
            elif price < bb_mid * 0.99: trend = "Bearish 🐻"

        return {"score": score, "label": label, "confidence": confidence, "trend": trend, "reasons": reasons}

    @staticmethod
    def generate_ai_summary(signal_result, price, predictions=None):
        lines = [
            "🤖 *ANALISA AI ISLM MONITOR V4*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"💰 *Harga:* Rp {price:,.0f}",
            f"📢 *Sinyal:* {signal_result['label']}",
            f"🎯 *Confidence:* {signal_result['confidence']*100:.0f}%",
            f"📈 *Trend:* {signal_result['trend']}",
            "", "📋 *REASONING:*"
        ]
        for r in signal_result.get("reasons", []):
            lines.append(f"  • {r}")
        if predictions:
            lines += ["", "🔮 *PREDIKSI:*"]
            for k, p in predictions.items():
                lines.append(f"  • {p['label']}: Rp {p['target']:,.0f} ({p['change_pct']:+.1f}%) {p['direction']}")
        lines.append(f"\n⏰ Update: {datetime.now().strftime('%H:%M:%S WIB')}")
        return "\n".join(lines)


# ============================================
# BACKWARD COMPATIBILITY ALIAS
# ============================================
class TelegramNotifier:
    """Legacy class — use TelegramBot from telegram_bot.py instead."""
    def __init__(self, token, chat_id=None):
        self.token = token
        self.chat_id = chat_id
    def send_message(self, message):
        if not self.chat_id: return False
        try:
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            threading.Thread(target=requests.post, args=(url,),
                             kwargs={"json": {"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"}},
                             daemon=True).start()
            return True
        except: return False
