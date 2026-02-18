"""
ISLM Monitor — Multi-Timeframe Analysis Engine V8
==================================================
Institutional-grade multi-TF analysis with:
  - 7 timeframes (1m, 5m, 15m, 1H, 4H, 1D, 1W)
  - Weighted TF scoring with regime detection
  - Market structure (HH/HL/LH/LL)
  - Breakout vs fake breakout detection
  - Bull trap / Dead cat bounce recognition
  - RSI/MACD divergence detection
  - VWAP, Volume Profile, Auto-Fibonacci
  - Volatility regime (expansion/contraction)
  - Volatility percentile (ATR vs 30d)
  - Probabilistic scenario modeling + confidence breakdown
"""

import numpy as np
import statistics
from datetime import datetime


class MultiTimeframeEngine:
    """Analyze across 7 timeframes with weighted institutional scoring."""

    TIMEFRAMES = {
        '1':   {'weight': 0.05, 'label': '1m',  'min_candles': 30},
        '5':   {'weight': 0.08, 'label': '5m',  'min_candles': 30},
        '15':  {'weight': 0.12, 'label': '15m', 'min_candles': 30},
        '60':  {'weight': 0.20, 'label': '1H',  'min_candles': 24},
        '240': {'weight': 0.20, 'label': '4H',  'min_candles': 20},
        '1D':  {'weight': 0.25, 'label': '1D',  'min_candles': 14},
        '1W':  {'weight': 0.10, 'label': '1W',  'min_candles': 8},
    }

    # TF weights shift based on regime
    REGIME_WEIGHTS = {
        'TRENDING': {'1': 0.03, '5': 0.05, '15': 0.10, '60': 0.20, '240': 0.22, '1D': 0.28, '1W': 0.12},
        'RANGING':  {'1': 0.08, '5': 0.12, '15': 0.18, '60': 0.22, '240': 0.18, '1D': 0.15, '1W': 0.07},
        'VOLATILE': {'1': 0.10, '5': 0.15, '15': 0.20, '60': 0.20, '240': 0.15, '1D': 0.13, '1W': 0.07},
    }

    @staticmethod
    def analyze_single_tf(candles, tf_label='15m'):
        """Analyze a single timeframe. Returns dict of indicators and signals."""
        if not candles or len(candles) < 10:
            return {'valid': False, 'tf': tf_label}

        closes = np.array([c['close'] for c in candles], dtype=float)
        highs = np.array([c['high'] for c in candles], dtype=float)
        lows = np.array([c['low'] for c in candles], dtype=float)
        volumes = np.array([c.get('vol', 0) for c in candles], dtype=float)

        result = {'valid': True, 'tf': tf_label, 'candles': len(candles)}

        # RSI
        result['rsi'] = MultiTimeframeEngine._rsi(closes)

        # MACD
        macd, signal, hist = MultiTimeframeEngine._macd(closes)
        result['macd'] = macd
        result['macd_signal'] = signal
        result['macd_hist'] = hist

        # EMA Cross
        if len(closes) >= 21:
            ema9 = MultiTimeframeEngine._ema(closes, 9)
            ema21 = MultiTimeframeEngine._ema(closes, 21)
            result['ema9'] = ema9
            result['ema21'] = ema21
            result['ema_cross'] = 'BULLISH' if ema9 > ema21 else 'BEARISH'
            result['ema_distance'] = round((ema9 - ema21) / ema21 * 100, 3) if ema21 > 0 else 0

        if len(closes) >= 200:
            ema50 = MultiTimeframeEngine._ema(closes, 50)
            ema200 = MultiTimeframeEngine._ema(closes, 200)
            result['ema50'] = ema50
            result['ema200'] = ema200
            result['golden_cross'] = ema50 > ema200
        elif len(closes) >= 50:
            result['ema50'] = MultiTimeframeEngine._ema(closes, 50)

        # Bollinger Bands
        if len(closes) >= 20:
            bb_mid = np.mean(closes[-20:])
            bb_std = np.std(closes[-20:])
            result['bb_upper'] = bb_mid + 2 * bb_std
            result['bb_mid'] = bb_mid
            result['bb_lower'] = bb_mid - 2 * bb_std
            result['bb_width'] = round((result['bb_upper'] - result['bb_lower']) / bb_mid * 100, 2) if bb_mid > 0 else 0
            result['bb_position'] = round((closes[-1] - result['bb_lower']) / max(result['bb_upper'] - result['bb_lower'], 0.001), 3)

        # ATR
        result['atr'] = MultiTimeframeEngine._atr(highs, lows, closes)

        # Stochastic RSI
        result['stoch_k'] = MultiTimeframeEngine._stochastic(closes)

        # VWAP (intraday only)
        if tf_label in ['1m', '5m', '15m', '1H']:
            result['vwap'] = MultiTimeframeEngine._vwap(closes, volumes)

        # Market Structure
        structure = MultiTimeframeEngine.detect_market_structure(candles)
        result.update(structure)

        # Divergence
        if len(closes) >= 14:
            rsi_vals = MultiTimeframeEngine._rsi_series(closes)
            result['rsi_divergence'] = MultiTimeframeEngine._detect_divergence(closes[-30:], rsi_vals[-30:]) if len(rsi_vals) >= 30 else 'NONE'

        # Volume analysis
        if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0:
            result['vol_sma20'] = float(np.mean(volumes[-20:]))
            result['vol_ratio'] = round(volumes[-1] / result['vol_sma20'], 2) if result['vol_sma20'] > 0 else 1.0
            result['vol_trend'] = 'RISING' if np.mean(volumes[-5:]) > np.mean(volumes[-20:]) else 'FALLING'

        # Compute TF signal: BULLISH / BEARISH / NEUTRAL
        bull_points = 0
        bear_points = 0

        rsi = result.get('rsi', 50)
        if rsi < 30: bull_points += 2  # oversold
        elif rsi < 40: bull_points += 1
        elif rsi > 70: bear_points += 2  # overbought
        elif rsi > 60: bear_points += 1

        if result.get('macd_hist', 0) > 0: bull_points += 1
        else: bear_points += 1

        if result.get('ema_cross') == 'BULLISH': bull_points += 2
        elif result.get('ema_cross') == 'BEARISH': bear_points += 2

        if result.get('structure_trend') == 'UPTREND': bull_points += 2
        elif result.get('structure_trend') == 'DOWNTREND': bear_points += 2

        bb_pos = result.get('bb_position', 0.5)
        if bb_pos < 0.2: bull_points += 1
        elif bb_pos > 0.8: bear_points += 1

        if result.get('rsi_divergence') == 'BULLISH': bull_points += 2
        elif result.get('rsi_divergence') == 'BEARISH': bear_points += 2

        total = bull_points + bear_points
        if total == 0:
            result['signal'] = 'NEUTRAL'
            result['signal_score'] = 0
        elif bull_points > bear_points:
            result['signal'] = 'BULLISH'
            result['signal_score'] = round(bull_points / total * 100, 1)
        else:
            result['signal'] = 'BEARISH'
            result['signal_score'] = round(-bear_points / total * 100, 1)

        return result

    @staticmethod
    def weighted_multi_tf_score(tf_results, regime='TRENDING'):
        """Compute weighted score across all timeframes. Returns -100 to +100."""
        weights = MultiTimeframeEngine.REGIME_WEIGHTS.get(regime, MultiTimeframeEngine.REGIME_WEIGHTS['TRENDING'])

        total_score = 0
        total_weight = 0
        tf_signals = {}
        bullish_count = 0
        bearish_count = 0

        for tf_key, tf_data in tf_results.items():
            if not tf_data.get('valid'):
                continue
            w = weights.get(tf_key, 0.1)
            score = tf_data.get('signal_score', 0)
            total_score += score * w
            total_weight += w
            tf_signals[tf_data.get('tf', tf_key)] = tf_data.get('signal', 'NEUTRAL')
            if tf_data.get('signal') == 'BULLISH': bullish_count += 1
            elif tf_data.get('signal') == 'BEARISH': bearish_count += 1

        weighted = total_score / max(total_weight, 0.01)
        total_tfs = bullish_count + bearish_count + (len(tf_results) - bullish_count - bearish_count)

        return {
            'weighted_score': round(weighted, 1),
            'regime': regime,
            'tf_alignment': f"{bullish_count}/{total_tfs} bullish",
            'bullish_tfs': bullish_count,
            'bearish_tfs': bearish_count,
            'neutral_tfs': total_tfs - bullish_count - bearish_count,
            'tf_signals': tf_signals,
            'consensus': 'STRONG_BUY' if weighted > 60 else
                         'BUY' if weighted > 25 else
                         'STRONG_SELL' if weighted < -60 else
                         'SELL' if weighted < -25 else 'NEUTRAL',
        }

    # ===== Regime Detection =====
    @staticmethod
    def detect_regime(candles_1d, candles_4h=None):
        """Detect market regime: TRENDING, RANGING, or VOLATILE."""
        if not candles_1d or len(candles_1d) < 14:
            return 'RANGING'

        closes = np.array([c['close'] for c in candles_1d[-30:]], dtype=float)

        # ADX-like measurement
        atr = MultiTimeframeEngine._atr(
            np.array([c['high'] for c in candles_1d[-30:]], dtype=float),
            np.array([c['low'] for c in candles_1d[-30:]], dtype=float),
            closes
        )

        # Price range vs ATR
        price_range = (max(closes) - min(closes)) / max(np.mean(closes), 1)
        volatility = (atr / max(np.mean(closes), 1)) * 100

        # EMA slope check
        if len(closes) >= 20:
            ema20 = MultiTimeframeEngine._ema(closes, min(20, len(closes)))
            ema_slope = (closes[-1] - ema20) / max(abs(ema20), 1)
        else:
            ema_slope = 0

        # Regime classification
        if volatility > 5 or price_range > 0.15:
            return 'VOLATILE'
        elif abs(ema_slope) > 0.02 or price_range > 0.08:
            return 'TRENDING'
        else:
            return 'RANGING'

    # ===== Market Structure =====
    @staticmethod
    def detect_market_structure(candles, lookback=20):
        """Detect HH/HL/LH/LL pattern for market structure."""
        if not candles or len(candles) < lookback:
            return {'structure_trend': 'UNKNOWN', 'swing_points': []}

        recent = candles[-lookback:]
        highs = [c['high'] for c in recent]
        lows = [c['low'] for c in recent]

        # Find swing highs and lows (local extrema)
        swing_highs = []
        swing_lows = []
        for i in range(2, len(recent) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append({'idx': i, 'price': highs[i]})
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append({'idx': i, 'price': lows[i]})

        structure = 'UNKNOWN'
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            hh = swing_highs[-1]['price'] > swing_highs[-2]['price']  # Higher High
            hl = swing_lows[-1]['price'] > swing_lows[-2]['price']    # Higher Low
            lh = swing_highs[-1]['price'] < swing_highs[-2]['price']  # Lower High
            ll = swing_lows[-1]['price'] < swing_lows[-2]['price']    # Lower Low

            if hh and hl:
                structure = 'UPTREND'
            elif lh and ll:
                structure = 'DOWNTREND'
            elif hh and ll:
                structure = 'EXPANDING'  # Volatile
            elif lh and hl:
                structure = 'CONTRACTING'  # Squeeze
            else:
                structure = 'SIDEWAYS'

        return {
            'structure_trend': structure,
            'swing_highs': swing_highs[-3:],
            'swing_lows': swing_lows[-3:],
            'last_high': swing_highs[-1]['price'] if swing_highs else None,
            'last_low': swing_lows[-1]['price'] if swing_lows else None,
        }

    # ===== Breakout Detection =====
    @staticmethod
    def detect_breakout(candles, support, resistance):
        """Detect breakout vs fakeout based on volume and close confirmation."""
        if not candles or len(candles) < 3 or not support or not resistance:
            return {'type': 'NONE'}

        last = candles[-1]
        prev = candles[-2]
        volumes = [c.get('vol', 0) for c in candles[-20:]]
        avg_vol = np.mean(volumes) if volumes else 0

        # Resistance breakout
        if last['close'] > resistance:
            vol_confirm = last.get('vol', 0) > avg_vol * 1.5
            close_confirm = last['close'] > resistance * 1.005  # >0.5% above
            if vol_confirm and close_confirm:
                return {'type': 'BREAKOUT_UP', 'level': resistance, 'confirmed': True, 'vol_ratio': round(last.get('vol', 0) / max(avg_vol, 1), 2)}
            else:
                return {'type': 'FAKEOUT_UP', 'level': resistance, 'confirmed': False, 'reason': 'weak_volume' if not vol_confirm else 'weak_close'}

        # Support breakdown
        if last['close'] < support:
            vol_confirm = last.get('vol', 0) > avg_vol * 1.5
            close_confirm = last['close'] < support * 0.995
            if vol_confirm and close_confirm:
                return {'type': 'BREAKOUT_DOWN', 'level': support, 'confirmed': True, 'vol_ratio': round(last.get('vol', 0) / max(avg_vol, 1), 2)}
            else:
                return {'type': 'FAKEOUT_DOWN', 'level': support, 'confirmed': False}

        return {'type': 'NONE'}

    # ===== Bull Trap / Dead Cat Bounce Detection =====
    @staticmethod
    def detect_bull_trap(candles, resistance=None):
        """Detect bull trap: price breaks above resistance then falls back."""
        if not candles or len(candles) < 5:
            return {'detected': False}

        recent = candles[-5:]
        # Bull trap pattern: price pushes above then reverses
        made_high = any(c['high'] > resistance for c in recent[:3]) if resistance else False
        fell_back = recent[-1]['close'] < recent[-3]['close']
        vol_declining = recent[-1].get('vol', 0) < recent[-3].get('vol', 1)

        if made_high and fell_back and vol_declining:
            return {
                'detected': True,
                'type': 'BULL_TRAP',
                'trap_high': max(c['high'] for c in recent),
                'current': recent[-1]['close'],
                'confidence': 0.7 if vol_declining else 0.5,
            }
        return {'detected': False}

    @staticmethod
    def detect_dead_cat_bounce(candles):
        """Detect dead cat bounce: small recovery in downtrend before continuing down."""
        if not candles or len(candles) < 10:
            return {'detected': False}

        recent = candles[-10:]
        closes = [c['close'] for c in recent]

        # Was there a sharp drop followed by a weak recovery?
        drop_phase = closes[:5]
        bounce_phase = closes[5:]

        drop_pct = (drop_phase[0] - min(drop_phase)) / max(drop_phase[0], 1)
        if drop_pct < 0.05:
            return {'detected': False}

        bounce_pct = (max(bounce_phase) - min(drop_phase)) / max(min(drop_phase), 1)
        # Dead cat: recovery < 50% of the drop
        fib_retrace = bounce_pct / max(drop_pct, 0.001)

        if 0.1 < fib_retrace < 0.5 and bounce_phase[-1] < bounce_phase[0]:
            return {
                'detected': True,
                'drop_pct': round(drop_pct * 100, 1),
                'bounce_pct': round(bounce_pct * 100, 1),
                'retrace_ratio': round(fib_retrace, 2),
                'confidence': 0.6 + (0.3 * (1 - fib_retrace)),
            }
        return {'detected': False}

    # ===== Volatility Regime =====
    @staticmethod
    def volatility_regime(candles, lookback=30):
        """Detect volatility expansion/contraction. Returns regime + percentile."""
        if not candles or len(candles) < lookback:
            return {'regime': 'UNKNOWN', 'percentile': 50}

        highs = np.array([c['high'] for c in candles[-lookback:]], dtype=float)
        lows = np.array([c['low'] for c in candles[-lookback:]], dtype=float)
        closes = np.array([c['close'] for c in candles[-lookback:]], dtype=float)

        atr14 = MultiTimeframeEngine._atr(highs, lows, closes, period=14)
        atr_history = []
        for i in range(14, len(closes)):
            a = MultiTimeframeEngine._atr(highs[:i+1], lows[:i+1], closes[:i+1], period=14)
            atr_history.append(a)

        if not atr_history:
            return {'regime': 'UNKNOWN', 'percentile': 50, 'atr': atr14}

        # Percentile: where current ATR sits vs 30-day history
        percentile = sum(1 for a in atr_history if a <= atr14) / len(atr_history) * 100

        # BB width for squeeze/expansion
        bb_width = None
        if len(closes) >= 20:
            bb_std = np.std(closes[-20:])
            bb_mid = np.mean(closes[-20:])
            bb_width = (bb_std * 4 / max(bb_mid, 1)) * 100

        regime = 'NORMAL'
        if percentile > 80:
            regime = 'EXPANSION'
        elif percentile < 20:
            regime = 'CONTRACTION'  # Squeeze — breakout likely

        return {
            'regime': regime,
            'percentile': round(percentile, 1),
            'atr': round(atr14, 2),
            'atr_mean': round(statistics.mean(atr_history), 2) if atr_history else 0,
            'bb_width': round(bb_width, 2) if bb_width else None,
            'is_squeeze': percentile < 15,
        }

    # ===== VWAP =====
    @staticmethod
    def _vwap(closes, volumes):
        """Volume Weighted Average Price."""
        if len(closes) == 0 or len(volumes) == 0:
            return 0
        cum_vol = np.cumsum(volumes)
        cum_pv = np.cumsum(closes * volumes)
        if cum_vol[-1] == 0:
            return float(closes[-1])
        return float(cum_pv[-1] / cum_vol[-1])

    # ===== Volume Profile =====
    @staticmethod
    def volume_profile(candles, bins=20):
        """Compute volume profile — volume at each price level."""
        if not candles or len(candles) < 10:
            return {'valid': False}

        prices = [c['close'] for c in candles]
        volumes = [c.get('vol', 0) for c in candles]
        price_min, price_max = min(prices), max(prices)
        if price_max == price_min:
            return {'valid': False}

        bin_size = (price_max - price_min) / bins
        profile = []
        for i in range(bins):
            lo = price_min + i * bin_size
            hi = lo + bin_size
            vol_at_level = sum(v for p, v in zip(prices, volumes) if lo <= p < hi)
            profile.append({'low': round(lo, 2), 'high': round(hi, 2), 'volume': round(vol_at_level, 2)})

        # Point of Control (POC) — highest volume level
        poc = max(profile, key=lambda x: x['volume'])

        # Value Area (70% of volume)
        sorted_profile = sorted(profile, key=lambda x: x['volume'], reverse=True)
        total_vol = sum(p['volume'] for p in profile)
        va_vol = 0
        va_levels = []
        for p in sorted_profile:
            va_vol += p['volume']
            va_levels.append(p)
            if va_vol >= total_vol * 0.7:
                break

        va_prices = [l['low'] for l in va_levels] + [l['high'] for l in va_levels]

        return {
            'valid': True,
            'poc': round((poc['low'] + poc['high']) / 2, 2),
            'poc_volume': poc['volume'],
            'value_area_high': round(max(va_prices), 2),
            'value_area_low': round(min(va_prices), 2),
            'profile': profile,
        }

    # ===== Auto Fibonacci =====
    @staticmethod
    def fibonacci_auto(candles, lookback=50):
        """Auto-detect swing high/low and compute Fibonacci levels."""
        if not candles or len(candles) < lookback:
            return {'valid': False}

        recent = candles[-lookback:]
        high = max(c['high'] for c in recent)
        low = min(c['low'] for c in recent)
        diff = high - low

        if diff <= 0:
            return {'valid': False}

        current = recent[-1]['close']
        is_uptrend = current > (high + low) / 2

        levels = {
            '0.0': high if is_uptrend else low,
            '0.236': high - diff * 0.236 if is_uptrend else low + diff * 0.236,
            '0.382': high - diff * 0.382 if is_uptrend else low + diff * 0.382,
            '0.5':   high - diff * 0.5   if is_uptrend else low + diff * 0.5,
            '0.618': high - diff * 0.618 if is_uptrend else low + diff * 0.618,
            '0.786': high - diff * 0.786 if is_uptrend else low + diff * 0.786,
            '1.0': low if is_uptrend else high,
        }

        # Current position relative to Fibonacci
        current_fib = (high - current) / diff if is_uptrend else (current - low) / diff

        return {
            'valid': True,
            'swing_high': round(high, 2),
            'swing_low': round(low, 2),
            'is_uptrend': is_uptrend,
            'levels': {k: round(v, 2) for k, v in levels.items()},
            'current_fib': round(current_fib, 3),
            'nearest_support': round(levels.get('0.618', low), 2),
            'nearest_resistance': round(levels.get('0.382', high), 2),
        }

    # ===== Probabilistic Scenarios =====
    @staticmethod
    def probabilistic_scenarios(candles, tf_results=None):
        """Generate 3 scenarios (bull/base/bear) with confidence scores."""
        if not candles or len(candles) < 20:
            return {'valid': False}

        closes = np.array([c['close'] for c in candles[-50:]], dtype=float)
        current = closes[-1]
        returns = np.diff(closes) / closes[:-1]

        if len(returns) < 10:
            return {'valid': False}

        mean_ret = float(np.mean(returns))
        std_ret = float(np.std(returns))

        # Scenario modeling
        bull_ret = mean_ret + std_ret
        base_ret = mean_ret
        bear_ret = mean_ret - std_ret

        # Confidence based on TF alignment
        bull_conf = 0.25
        base_conf = 0.50
        bear_conf = 0.25

        if tf_results:
            bullish_tfs = sum(1 for r in tf_results.values() if r.get('signal') == 'BULLISH')
            bearish_tfs = sum(1 for r in tf_results.values() if r.get('signal') == 'BEARISH')
            total = max(bullish_tfs + bearish_tfs, 1)

            if bullish_tfs > bearish_tfs:
                bull_conf = 0.35 + 0.15 * (bullish_tfs / total)
                bear_conf = 0.15
                base_conf = 1.0 - bull_conf - bear_conf
            elif bearish_tfs > bullish_tfs:
                bear_conf = 0.35 + 0.15 * (bearish_tfs / total)
                bull_conf = 0.15
                base_conf = 1.0 - bull_conf - bear_conf

        scenarios = {
            'bull': {
                'label': '🟢 Bullish',
                'target': round(current * (1 + bull_ret * 5), 2),
                'change_pct': round(bull_ret * 500, 1),
                'probability': round(bull_conf * 100, 1),
            },
            'base': {
                'label': '🟡 Base Case',
                'target': round(current * (1 + base_ret * 5), 2),
                'change_pct': round(base_ret * 500, 1),
                'probability': round(base_conf * 100, 1),
            },
            'bear': {
                'label': '🔴 Bearish',
                'target': round(current * (1 + bear_ret * 5), 2),
                'change_pct': round(bear_ret * 500, 1),
                'probability': round(bear_conf * 100, 1),
            },
        }

        # Confidence breakdown (5 factors)
        confidence_factors = {
            'trend_alignment': round(abs(bullish_tfs - bearish_tfs) / max(total, 1) * 100, 1) if tf_results else 50,
            'momentum': round(min(100, abs(mean_ret / max(std_ret, 0.001)) * 50), 1),
            'volatility': round(max(0, 100 - std_ret * 1000), 1),
            'volume_support': 50,  # Updated by caller with real data
            'structure': 50,  # Updated by caller with structure data
        }

        return {
            'valid': True,
            'current': current,
            'scenarios': scenarios,
            'dominant': 'bull' if bull_conf > bear_conf and bull_conf > base_conf else
                        'bear' if bear_conf > bull_conf else 'base',
            'confidence_factors': confidence_factors,
            'overall_confidence': round(max(bull_conf, base_conf, bear_conf) * 100, 1),
        }

    # ===== Helper functions =====
    @staticmethod
    def _rsi(closes, period=14):
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - 100 / (1 + rs), 1)

    @staticmethod
    def _rsi_series(closes, period=14):
        """Return RSI for each point (for divergence detection)."""
        rsi_vals = []
        for i in range(period, len(closes)):
            rsi_vals.append(MultiTimeframeEngine._rsi(closes[:i+1], period))
        return rsi_vals

    @staticmethod
    def _macd(closes, fast=12, slow=26, sig=9):
        if len(closes) < slow + sig:
            return 0, 0, 0
        ema_fast = MultiTimeframeEngine._ema(closes, fast)
        ema_slow = MultiTimeframeEngine._ema(closes, slow)
        macd_line = ema_fast - ema_slow
        # Simplified signal: use last sig values' average
        macd_series = []
        for i in range(slow, len(closes)):
            ef = MultiTimeframeEngine._ema(closes[:i+1], fast)
            es = MultiTimeframeEngine._ema(closes[:i+1], slow)
            macd_series.append(ef - es)
        signal_line = np.mean(macd_series[-sig:]) if len(macd_series) >= sig else macd_line
        hist = macd_line - signal_line
        return round(macd_line, 4), round(signal_line, 4), round(hist, 4)

    @staticmethod
    def _ema(data, period):
        if len(data) == 0:
            return 0
        if len(data) < period:
            return float(np.mean(data))
        k = 2 / (period + 1)
        ema = float(data[0])
        for val in data[1:]:
            ema = val * k + ema * (1 - k)
        return round(ema, 4)

    @staticmethod
    def _atr(highs, lows, closes, period=14):
        if len(closes) < 2:
            return 0
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        return round(float(np.mean(trs[-period:])), 4) if trs else 0

    @staticmethod
    def _stochastic(closes, period=14):
        if len(closes) < period:
            return 50.0
        window = closes[-period:]
        lo = min(window)
        hi = max(window)
        if hi == lo:
            return 50.0
        k = (closes[-1] - lo) / (hi - lo) * 100
        return round(k, 1)

    @staticmethod
    def _detect_divergence(prices, indicator_vals):
        """Detect bullish/bearish divergence between price and indicator."""
        if len(prices) < 10 or len(indicator_vals) < 10:
            return 'NONE'

        # Compare last 2 swing points
        price_start, price_end = prices[0], prices[-1]
        ind_start, ind_end = indicator_vals[0], indicator_vals[-1]

        # Bullish divergence: price makes lower low, indicator makes higher low
        if price_end < price_start and ind_end > ind_start:
            return 'BULLISH'
        # Bearish divergence: price makes higher high, indicator makes lower high
        if price_end > price_start and ind_end < ind_start:
            return 'BEARISH'

        return 'NONE'


# ===== Market Session Awareness =====
class MarketSessionDetector:
    """Detect current market session for volatility behavior."""

    @staticmethod
    def get_current_session():
        """Returns current crypto market session based on UTC time."""
        utc_hour = datetime.utcnow().hour

        if 0 <= utc_hour < 8:
            return {
                'session': 'ASIA',
                'label': '🌏 Asia Session',
                'typical_vol': 'LOW-MEDIUM',
                'description': 'Tokyo/Singapore/HK active. Lower volatility.',
            }
        elif 8 <= utc_hour < 14:
            return {
                'session': 'EUROPE',
                'label': '🌍 Europe Session',
                'typical_vol': 'MEDIUM',
                'description': 'London active. Moderate volatility.',
            }
        elif 14 <= utc_hour < 21:
            return {
                'session': 'US',
                'label': '🌎 US Session',
                'typical_vol': 'HIGH',
                'description': 'NY active. Highest volume period.',
            }
        else:
            return {
                'session': 'OVERLAP',
                'label': '🌐 Late US / Early Asia',
                'typical_vol': 'MEDIUM-HIGH',
                'description': 'Transition period. Watch for moves.',
            }
