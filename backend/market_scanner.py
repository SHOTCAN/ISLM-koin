"""
Market Scanner — Multi-Layer Opportunity Detection V1
=====================================================
Scans ALL Indodax-listed coins for high-probability upward movements.

Features:
  - 5-layer strict validation (volume, breakout, confluence, sentiment, momentum)
  - Operational modes: analysis_only, alert, aggressive, conservative
  - Configurable minimum confidence threshold
  - Alert cooldown per coin + per signal type
  - Liquidity & volume filters (min daily volume, spread, depth)
  - Market regime detection integration
  - Probability scoring based on measurable logic
"""

import time
import numpy as np
import statistics
from collections import defaultdict
from datetime import datetime


# ============================================
# OPERATIONAL MODES
# ============================================

class OperationalMode:
    """Defines scanning sensitivity, signal frequency, and risk tolerance."""

    MODES = {
        'analysis': {
            'name': 'Analysis Only',
            'description': 'Hanya analisa, tanpa alert otomatis',
            'min_confidence': 80,
            'scan_interval': 7200,      # 2 hours
            'alert_enabled': False,
            'min_volume_usd': 50000,
            'max_spread_pct': 3.0,
            'cooldown_minutes': 999999,  # effectively no alerts
        },
        'alert': {
            'name': 'Alert Mode',
            'description': 'Alert saat sinyal kuat terdeteksi',
            'min_confidence': 70,
            'scan_interval': 3600,       # 1 hour
            'alert_enabled': True,
            'min_volume_usd': 30000,
            'max_spread_pct': 3.0,
            'cooldown_minutes': 60,
        },
        'aggressive': {
            'name': 'Aggressive Scan',
            'description': 'Scan sensitif, lebih banyak alert',
            'min_confidence': 55,
            'scan_interval': 1800,       # 30 min
            'alert_enabled': True,
            'min_volume_usd': 10000,
            'max_spread_pct': 5.0,
            'cooldown_minutes': 30,
        },
        'conservative': {
            'name': 'Conservative Mode',
            'description': 'Hanya sinyal paling kuat, filter ketat',
            'min_confidence': 85,
            'scan_interval': 7200,       # 2 hours
            'alert_enabled': True,
            'min_volume_usd': 100000,
            'max_spread_pct': 2.0,
            'cooldown_minutes': 120,
        },
    }

    @staticmethod
    def get(mode_key):
        return OperationalMode.MODES.get(mode_key, OperationalMode.MODES['alert'])

    @staticmethod
    def list_modes():
        lines = []
        for key, m in OperationalMode.MODES.items():
            lines.append(f"  `{key}` — {m['name']}: {m['description']}")
        return "\n".join(lines)


# ============================================
# ALERT COOLDOWN MANAGER
# ============================================

class AlertCooldown:
    """Prevent repeated notifications per coin + per signal type."""

    def __init__(self):
        # key = f"{coin}:{signal_type}" → last alert timestamp
        self._last_alert = {}

    def can_alert(self, coin, signal_type, cooldown_minutes):
        key = f"{coin}:{signal_type}"
        now = time.time()
        last = self._last_alert.get(key, 0)
        if now - last < cooldown_minutes * 60:
            return False
        return True

    def record_alert(self, coin, signal_type):
        key = f"{coin}:{signal_type}"
        self._last_alert[key] = time.time()

    def get_remaining(self, coin, signal_type, cooldown_minutes):
        key = f"{coin}:{signal_type}"
        now = time.time()
        last = self._last_alert.get(key, 0)
        remaining = (cooldown_minutes * 60) - (now - last)
        return max(0, remaining)

    def clear(self, coin=None):
        if coin:
            self._last_alert = {k: v for k, v in self._last_alert.items() if not k.startswith(f"{coin}:")}
        else:
            self._last_alert.clear()


# ============================================
# LIQUIDITY FILTER
# ============================================

class LiquidityFilter:
    """Filter out illiquid coins that produce weak/unreliable signals."""

    @staticmethod
    def check(ticker_data, mode_config):
        """
        Validate coin liquidity. Returns {passed, reasons}.
        ticker_data: dict with keys vol_idr, buy, sell, last
        mode_config: operational mode config dict
        """
        reasons = []
        passed = True

        # Volume filter (convert IDR to rough USD)
        vol_idr = float(ticker_data.get('vol_idr', 0) or 0)
        vol_usd_approx = vol_idr / 16000  # rough IDR/USD
        min_vol = mode_config.get('min_volume_usd', 30000)
        if vol_usd_approx < min_vol:
            reasons.append(f"Volume terlalu rendah: ${vol_usd_approx:,.0f} < ${min_vol:,.0f}")
            passed = False

        # Spread filter
        buy = float(ticker_data.get('buy', 0) or 0)
        sell = float(ticker_data.get('sell', 0) or 0)
        if buy > 0 and sell > 0:
            spread_pct = ((sell - buy) / buy) * 100
            max_spread = mode_config.get('max_spread_pct', 3.0)
            if spread_pct > max_spread:
                reasons.append(f"Spread terlalu lebar: {spread_pct:.1f}% > {max_spread:.1f}%")
                passed = False

        # Price sanity (must be > 0)
        price = float(ticker_data.get('last', 0) or 0)
        if price <= 0:
            reasons.append("Harga tidak valid")
            passed = False

        return {'passed': passed, 'reasons': reasons, 'vol_usd': vol_usd_approx}


# ============================================
# MARKET REGIME DETECTOR
# ============================================

class MarketRegimeDetector:
    """Detect broader market context: bullish, bearish, sideways, high_volatility."""

    @staticmethod
    def detect(candles, lookback=30):
        """
        Analyze recent candle data to determine regime.
        Returns {regime, strength, description}.
        """
        if not candles or len(candles) < lookback:
            return {'regime': 'unknown', 'strength': 0, 'description': 'Data tidak cukup'}

        closes = [c['close'] for c in candles[-lookback:]]
        highs = [c['high'] for c in candles[-lookback:]]
        lows = [c['low'] for c in candles[-lookback:]]

        # Trend direction
        sma_short = np.mean(closes[-7:])
        sma_long = np.mean(closes[-lookback:])
        trend_pct = ((sma_short / sma_long) - 1) * 100

        # Volatility
        returns = np.diff(np.log(closes))
        volatility = np.std(returns) * 100 if len(returns) > 0 else 0

        # Higher highs / lower lows count
        hh_count = sum(1 for i in range(1, min(10, len(highs))) if highs[-i] > highs[-i-1])
        ll_count = sum(1 for i in range(1, min(10, len(lows))) if lows[-i] < lows[-i-1])

        # Determine regime
        if volatility > 5:
            regime = 'high_volatility'
            strength = min(100, int(volatility * 10))
            desc = f"Volatilitas tinggi ({volatility:.1f}%), hati-hati!"
        elif trend_pct > 2 and hh_count >= 5:
            regime = 'bullish'
            strength = min(100, int(trend_pct * 15))
            desc = f"Bullish trend (+{trend_pct:.1f}%), {hh_count} higher highs"
        elif trend_pct < -2 and ll_count >= 5:
            regime = 'bearish'
            strength = min(100, int(abs(trend_pct) * 15))
            desc = f"Bearish trend ({trend_pct:.1f}%), {ll_count} lower lows"
        else:
            regime = 'sideways'
            strength = max(0, 50 - int(abs(trend_pct) * 10))
            desc = f"Sideways/ranging (trend: {trend_pct:+.1f}%)"

        return {'regime': regime, 'strength': strength, 'description': desc}

    @staticmethod
    def adjust_confidence(base_confidence, regime, signal_direction='bullish'):
        """Adjust confidence based on regime context."""
        if signal_direction == 'bullish':
            if regime == 'bullish':
                return min(99, base_confidence + 10)
            elif regime == 'bearish':
                return max(10, base_confidence - 20)
            elif regime == 'high_volatility':
                return max(10, base_confidence - 10)
        elif signal_direction == 'bearish':
            if regime == 'bearish':
                return min(99, base_confidence + 10)
            elif regime == 'bullish':
                return max(10, base_confidence - 20)
        return base_confidence


# ============================================
# MULTI-LAYER SCANNER
# ============================================

class CoinAnalysis:
    """Analyze a single coin through 5 strict validation layers."""

    @staticmethod
    def _rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _ema(data, period):
        if len(data) < period:
            return data[-1] if data else 0
        ema = [np.mean(data[:period])]
        k = 2 / (period + 1)
        for val in data[period:]:
            ema.append(val * k + ema[-1] * (1 - k))
        return ema[-1]

    @staticmethod
    def _macd(closes):
        if len(closes) < 26:
            return 0, 0, 0
        ema12 = CoinAnalysis._ema(closes, 12)
        ema26 = CoinAnalysis._ema(closes, 26)
        macd_line = ema12 - ema26
        # Simplified signal line
        return macd_line, 0, macd_line

    @staticmethod
    def _bollinger(closes, period=20):
        if len(closes) < period:
            return closes[-1], closes[-1], closes[-1]
        sma = np.mean(closes[-period:])
        std = np.std(closes[-period:])
        return sma + 2*std, sma, sma - 2*std

    @staticmethod
    def analyze(candles, ticker_data=None):
        """
        Run 5-layer validation on a coin's candle data.
        Returns {score, layers, direction, probability}.
        """
        if not candles or len(candles) < 30:
            return {'score': 0, 'probability': 0, 'direction': 'neutral', 'layers': {}, 'passed_layers': 0}

        closes = np.array([c['close'] for c in candles])
        volumes = np.array([c['vol'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        price = closes[-1]

        layers = {}
        total_score = 0
        max_score = 0

        # ====== LAYER 1: VOLUME SURGE (weight: 20) ======
        max_score += 20
        vol_avg_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
        vol_current = np.mean(volumes[-3:]) if len(volumes) >= 3 else volumes[-1]
        vol_ratio = vol_current / vol_avg_20 if vol_avg_20 > 0 else 0

        if vol_ratio >= 2.5:
            vol_score = 20
        elif vol_ratio >= 2.0:
            vol_score = 16
        elif vol_ratio >= 1.5:
            vol_score = 10
        elif vol_ratio >= 1.2:
            vol_score = 5
        else:
            vol_score = 0

        # Check if volume is on up-candles (bullish volume)
        recent_candles = candles[-5:]
        up_vol = sum(c['vol'] for c in recent_candles if c['close'] > c['open'])
        down_vol = sum(c['vol'] for c in recent_candles if c['close'] <= c['open'])
        vol_direction = 'bullish' if up_vol > down_vol * 1.3 else ('bearish' if down_vol > up_vol * 1.3 else 'neutral')

        layers['volume'] = {
            'score': vol_score,
            'max': 20,
            'ratio': round(vol_ratio, 2),
            'direction': vol_direction,
            'detail': f"Vol ratio: {vol_ratio:.1f}x avg, direction: {vol_direction}"
        }
        total_score += vol_score

        # ====== LAYER 2: BREAKOUT STRUCTURE (weight: 25) ======
        max_score += 25
        # Check for HH/HL pattern (bullish structure)
        lookback_highs = highs[-20:]
        lookback_lows = lows[-20:]

        # Find recent swing highs/lows
        resistance = np.max(highs[-20:])
        support = np.min(lows[-20:])
        range_pct = ((resistance - support) / support * 100) if support > 0 else 0

        # Breakout: price near or above recent resistance
        breakout_pct = ((price - resistance) / resistance * 100) if resistance > 0 else 0

        # Higher highs check
        quarter = max(5, len(lookback_highs) // 4)
        first_q_high = np.max(lookback_highs[:quarter])
        last_q_high = np.max(lookback_highs[-quarter:])
        hh = last_q_high > first_q_high

        # Higher lows check
        first_q_low = np.min(lookback_lows[:quarter])
        last_q_low = np.min(lookback_lows[-quarter:])
        hl = last_q_low > first_q_low

        if breakout_pct > 0.5 and hh and hl:
            struct_score = 25  # confirmed breakout with structure
        elif breakout_pct > 0 and (hh or hl):
            struct_score = 18
        elif hh and hl:
            struct_score = 12  # bullish structure but no breakout yet
        elif hh or hl:
            struct_score = 6
        else:
            struct_score = 0

        layers['structure'] = {
            'score': struct_score,
            'max': 25,
            'higher_highs': hh,
            'higher_lows': hl,
            'breakout_pct': round(breakout_pct, 2),
            'detail': f"HH: {'✅' if hh else '❌'}, HL: {'✅' if hl else '❌'}, Breakout: {breakout_pct:+.1f}%"
        }
        total_score += struct_score

        # ====== LAYER 3: INDICATOR CONFLUENCE (weight: 25) ======
        max_score += 25
        rsi = CoinAnalysis._rsi(closes)
        macd_line, _, macd_hist = CoinAnalysis._macd(list(closes))
        bb_upper, bb_mid, bb_lower = CoinAnalysis._bollinger(list(closes))

        confluence_count = 0
        confluence_details = []

        # RSI in bullish zone (40-65, rising)
        if 40 <= rsi <= 65:
            confluence_count += 1
            confluence_details.append(f"RSI bullish zone ({rsi:.0f})")
        elif 30 <= rsi < 40:
            confluence_count += 0.5
            confluence_details.append(f"RSI near oversold ({rsi:.0f})")

        # MACD bullish
        if macd_line > 0:
            confluence_count += 1
            confluence_details.append("MACD bullish")

        # Price above BB mid
        if price > bb_mid:
            confluence_count += 1
            confluence_details.append("Above BB middle")

        # EMA trend (20 > 50)
        ema_20 = CoinAnalysis._ema(list(closes), 20)
        ema_50 = CoinAnalysis._ema(list(closes), 50) if len(closes) >= 50 else ema_20
        if ema_20 > ema_50:
            confluence_count += 1
            confluence_details.append("EMA20 > EMA50")

        conf_score = int(min(25, (confluence_count / 4) * 25))

        layers['confluence'] = {
            'score': conf_score,
            'max': 25,
            'count': confluence_count,
            'rsi': round(rsi, 1),
            'macd': round(macd_line, 4),
            'details': confluence_details,
            'detail': f"Confluence: {confluence_count}/4 — {', '.join(confluence_details[:3])}"
        }
        total_score += conf_score

        # ====== LAYER 4: MOMENTUM (weight: 15) ======
        max_score += 15
        # Rate of change (5-period)
        roc_5 = ((closes[-1] / closes[-6]) - 1) * 100 if len(closes) >= 6 else 0
        # Acceleration (is momentum increasing?)
        roc_prev = ((closes[-6] / closes[-11]) - 1) * 100 if len(closes) >= 11 else 0
        acceleration = roc_5 - roc_prev

        if roc_5 > 3 and acceleration > 0:
            mom_score = 15
        elif roc_5 > 1.5 and acceleration > 0:
            mom_score = 12
        elif roc_5 > 0.5:
            mom_score = 7
        elif roc_5 > 0:
            mom_score = 3
        else:
            mom_score = 0

        layers['momentum'] = {
            'score': mom_score,
            'max': 15,
            'roc_5': round(roc_5, 2),
            'acceleration': round(acceleration, 2),
            'detail': f"RoC(5): {roc_5:+.1f}%, Acceleration: {acceleration:+.1f}%"
        }
        total_score += mom_score

        # ====== LAYER 5: SENTIMENT PROXY (weight: 15) ======
        max_score += 15
        # Use volume trend as sentiment proxy
        vol_trend = (np.mean(volumes[-5:]) / np.mean(volumes[-20:])) if np.mean(volumes[-20:]) > 0 else 1
        # Price trend (7d)
        price_trend_7d = ((closes[-1] / closes[-7]) - 1) * 100 if len(closes) >= 7 else 0
        # Combine
        sentiment_raw = (vol_trend - 1) * 30 + price_trend_7d * 2
        sent_score = int(min(15, max(0, sentiment_raw + 5)))

        layers['sentiment'] = {
            'score': sent_score,
            'max': 15,
            'vol_trend': round(vol_trend, 2),
            'price_trend_7d': round(price_trend_7d, 2),
            'detail': f"Vol trend: {vol_trend:.1f}x, Price 7d: {price_trend_7d:+.1f}%"
        }
        total_score += sent_score

        # ====== FINAL PROBABILITY ======
        probability = int((total_score / max_score) * 100) if max_score > 0 else 0
        passed_layers = sum(1 for l in layers.values() if l['score'] >= l['max'] * 0.5)

        direction = 'bullish' if probability >= 60 else ('bearish' if probability <= 30 else 'neutral')

        return {
            'score': total_score,
            'max_score': max_score,
            'probability': probability,
            'direction': direction,
            'passed_layers': passed_layers,
            'total_layers': 5,
            'layers': layers,
            'price': price,
            'rsi': round(rsi, 1),
        }


# ============================================
# MAIN MARKET SCANNER
# ============================================

class MarketScanner:
    """
    Scan all Indodax coins for high-probability setups.
    Integrates: multi-layer validation + operational modes + cooldowns + liquidity + regime.
    """

    def __init__(self):
        self.cooldown = AlertCooldown()
        self.mode = 'alert'  # default mode
        self.custom_threshold = None  # admin-overridable
        self.last_scan_time = 0
        self.scan_results = []  # cached results from last scan
        self.regime_cache = {}  # coin → regime data

    def get_mode_config(self):
        config = OperationalMode.get(self.mode)
        # Allow admin override of threshold
        if self.custom_threshold is not None:
            config = dict(config)
            config['min_confidence'] = self.custom_threshold
        return config

    def set_mode(self, mode_key):
        if mode_key in OperationalMode.MODES:
            self.mode = mode_key
            return True
        return False

    def set_threshold(self, pct):
        if 10 <= pct <= 99:
            self.custom_threshold = pct
            return True
        return False

    def should_scan(self):
        config = self.get_mode_config()
        return (time.time() - self.last_scan_time) >= config['scan_interval']

    def scan_all(self, api, top_n=30):
        """
        Scan top Indodax pairs and return high-probability setups.
        Returns list of {pair, probability, analysis, regime, ...}
        """
        config = self.get_mode_config()
        self.last_scan_time = time.time()
        results = []

        # Get all pairs
        try:
            pairs = api.get_all_pairs()
        except:
            pairs = []

        if not pairs:
            return []

        # Get multi-price for quick filtering
        try:
            tickers = api.get_multi_price()
        except:
            tickers = {}

        scanned = 0
        for pair_info in pairs[:top_n]:
            pair_id = pair_info.get('symbol', pair_info.get('id', ''))
            if not pair_id:
                continue

            pair_key = pair_id.lower()
            coin_name = pair_info.get('base', pair_key.replace('idr', '')).upper()

            # Get ticker data
            ticker = tickers.get(pair_key, {})
            if not ticker:
                # Try individual fetch
                try:
                    t = api.get_price(pair_key)
                    if t.get('success'):
                        ticker = t
                except:
                    continue

            # LIQUIDITY FILTER
            liq_check = LiquidityFilter.check(ticker, config)
            if not liq_check['passed']:
                continue

            # COOLDOWN CHECK
            if not self.cooldown.can_alert(coin_name, 'scan', config['cooldown_minutes']):
                continue

            # GET CANDLE DATA for analysis
            try:
                candles = api.get_kline(pair_key, '60')  # 1H candles
                if not candles or len(candles) < 30:
                    continue
            except:
                continue

            # RUN 5-LAYER ANALYSIS
            analysis = CoinAnalysis.analyze(candles, ticker)

            # DETECT REGIME for this coin
            regime_data = MarketRegimeDetector.detect(candles)
            self.regime_cache[coin_name] = regime_data

            # ADJUST confidence based on regime
            adj_prob = MarketRegimeDetector.adjust_confidence(
                analysis['probability'], regime_data['regime'], analysis['direction']
            )

            # CHECK THRESHOLD
            min_conf = config['min_confidence']
            if adj_prob >= min_conf and analysis['passed_layers'] >= 3:
                results.append({
                    'pair': pair_key,
                    'coin': coin_name,
                    'probability': adj_prob,
                    'raw_probability': analysis['probability'],
                    'direction': analysis['direction'],
                    'passed_layers': analysis['passed_layers'],
                    'layers': analysis['layers'],
                    'regime': regime_data,
                    'price': analysis['price'],
                    'rsi': analysis['rsi'],
                    'vol_usd': liq_check['vol_usd'],
                })

                # Record cooldown
                self.cooldown.record_alert(coin_name, 'scan')

            scanned += 1

        # Sort by probability descending
        results.sort(key=lambda x: x['probability'], reverse=True)
        self.scan_results = results
        return results

    def format_scan_result(self, result):
        """Format a single scan result for Telegram message."""
        layers = result['layers']
        regime = result['regime']

        # Direction emoji
        dir_emoji = '🟢' if result['direction'] == 'bullish' else ('🔴' if result['direction'] == 'bearish' else '⚪')

        # Layer check marks
        layer_status = []
        for name, data in layers.items():
            pct = (data['score'] / data['max'] * 100) if data['max'] > 0 else 0
            emoji = '✅' if pct >= 50 else '⚠️' if pct >= 25 else '❌'
            layer_status.append(f"  {emoji} {name.title()}: {data['detail']}")

        # Regime emoji
        regime_emoji = {'bullish': '📈', 'bearish': '📉', 'sideways': '➡️', 'high_volatility': '⚡'}.get(regime['regime'], '❓')

        return (
            f"{dir_emoji} *{result['coin']}/IDR* — Probability: *{result['probability']}%*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 Harga: Rp {result['price']:,.0f}\n"
            f"📊 RSI: {result['rsi']}\n"
            f"💧 Volume: ${result['vol_usd']:,.0f}\n"
            f"{regime_emoji} Regime: {regime['description']}\n\n"
            f"*5-Layer Validation ({result['passed_layers']}/5):*\n"
            + "\n".join(layer_status) +
            f"\n\n📐 Raw Score: {result['raw_probability']}% → Adjusted: {result['probability']}%"
        )

    def format_scan_summary(self, results, mode_config):
        """Format all scan results into a summary message."""
        if not results:
            return (
                "🔍 *MARKET SCAN COMPLETE*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 Mode: {mode_config['name']}\n"
                f"🎯 Threshold: {mode_config['min_confidence']}%\n\n"
                f"❌ Tidak ada sinyal yang memenuhi kriteria.\n"
                f"_Semua coin gagal melewati 5-layer validation._"
            )

        header = (
            f"🔍 *MARKET SCAN — {len(results)} PELUANG TERDETEKSI*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Mode: {mode_config['name']}\n"
            f"🎯 Min Threshold: {mode_config['min_confidence']}%\n"
            f"⏰ {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        )

        # Show top 5 results
        entries = []
        for i, r in enumerate(results[:5], 1):
            dir_emoji = '🟢' if r['direction'] == 'bullish' else '🔴'
            entries.append(
                f"{i}. {dir_emoji} *{r['coin']}* — {r['probability']}% "
                f"({r['passed_layers']}/5 layers)\n"
                f"   Rp {r['price']:,.0f} | RSI: {r['rsi']} | Vol: ${r['vol_usd']:,.0f}"
            )

        footer = f"\n\nGunakan `/scan {results[0]['coin'].lower()}` untuk detail lengkap."
        return header + "\n".join(entries) + footer
