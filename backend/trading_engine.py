"""
ISLM Autonomous Trading Engine V2 (Fully Intelligent)
======================================================
Central orchestrator connecting ALL analysis modules:
  CoinScanner → SignalPrecision → MultiTF → QualityGate
  → ManipulationCheck → RiskManager → Execute → Monitor → Learn

V2 Enhancements over V1:
  - Deep integration with signal_precision.py (5-factor probabilistic)
  - Multi-timeframe confirmation via multi_tf_engine.py
  - Manipulation detection gate via adaptive_engine.py
  - Market regime awareness for strategy adaptation
  - ATR-based dynamic stop loss (not fixed %)
  - Post-trade self-learning + weight adjustment
  - Slippage protection + spread validation
  - Daily auto-notification at midnight
  - Robust error recovery in autonomous loop
"""

import time
import uuid
import json
import os
import threading
import traceback
from datetime import datetime, timedelta


class TradingConfig:
    """All configurable trading parameters — ULTRA-SAFE MODE."""

    # === Capital Management ===
    INITIAL_CAPITAL = 50_000           # Rp50,000 starting capital
    MAX_RISK_PER_TRADE_PCT = 3.0       # Max 3% of capital per trade (TIGHT)
    MAX_EXPOSURE_TOTAL_PCT = 20.0      # Max 20% of capital in open positions
    MAX_DAILY_LOSS_IDR = 5_000         # STOP if daily loss > Rp5k (very tight)
    MAX_DAILY_TRADES = 5               # Max 5 trades/day
    MAX_OPEN_POSITIONS = 2             # Max 2 simultaneous positions

    # === Signal Thresholds (VERY STRICT — only trade when very confident) ===
    MIN_CONFIDENCE_TO_TRADE = 75       # Min 75% confidence to open trade
    MIN_SCAN_SCORE = 60                # Min scanner score
    MIN_SIGNAL_AGREEMENT = 3           # Min factors agreeing (of 5+)
    CONFIDENCE_BOOST_MULTI_TF = 5      # Boost if multi-TF aligns

    # === Stop Loss / Take Profit (TIGHT — protect capital) ===
    DEFAULT_SL_PCT = 2.0               # Tight 2% stop loss
    DEFAULT_TP_PCT = 4.0               # 4% take profit (2:1 RR)
    ATR_SL_MULTIPLIER = 1.2            # Tight ATR-based SL
    ATR_TP_MULTIPLIER = 2.5            # ATR-based TP (2:1 ratio)
    TRAILING_STOP_ACTIVATION_PCT = 1.2 # Activate trailing early at +1.2%
    TRAILING_STOP_DISTANCE_PCT = 0.8   # Trail tightly by 0.8%

    # === Partial Take-Profit (lock in profits fast) ===
    PARTIAL_TP_LEVELS = [
        (1.5, 0.30),  # At +1.5%, secure 30%
        (3.0, 0.40),  # At +3.0%, secure 40%
        (5.0, 0.30),  # At +5.0%, close remaining
    ]

    # === Timing ===
    SCAN_INTERVAL_SECONDS = 45         # Scan every 45s
    POSITION_CHECK_SECONDS = 8         # Check positions every 8s (fast reaction)
    DAILY_REPORT_HOUR_UTC = 17         # 00:00 WIB
    COOLDOWN_AFTER_LOSS_SECONDS = 600  # 10 min cooldown after loss
    COOLDOWN_AFTER_3_LOSSES = 3600     # 1 HOUR cooldown after 3 losses

    # === Capital Scaling Gates ===
    SCALE_REQUIREMENTS = {
        100_000: (15, 60.0, 1.5),      # Need 60% WR + 1.5 PF
        200_000: (30, 62.0, 1.7),
        500_000: (60, 65.0, 2.0),
    }

    # === Filters (STRICT) ===
    MAX_SPREAD_PCT = 1.5               # Only liquid coins
    MIN_VOLUME_IDR = 20_000_000        # Min Rp20M volume (liquid only)
    MAX_SLIPPAGE_PCT = 0.5             # Very tight slippage control


class CapitalManager:
    """Dynamic capital management with performance-based scaling."""

    def __init__(self, initial_capital=None, data_dir='data'):
        os.makedirs(data_dir, exist_ok=True)
        self._filepath = os.path.join(data_dir, 'capital_state.json')
        self._state = self._load()
        if initial_capital and self._state.get('capital', 0) == 0:
            self._state['capital'] = initial_capital
            self._state['initial_capital'] = initial_capital
            self._save()

    def _load(self):
        try:
            if os.path.exists(self._filepath):
                with open(self._filepath, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'capital': 0, 'initial_capital': 0, 'allocated': 0,
                'scale_level': 0, 'scale_history': [], 'last_updated': None}

    def _save(self):
        self._state['last_updated'] = datetime.utcnow().isoformat()
        try:
            with open(self._filepath, 'w') as f:
                json.dump(self._state, f, indent=2)
        except Exception as e:
            print(f"[Capital] Save error: {e}")

    @property
    def capital(self):
        return self._state.get('capital', 0)

    @property
    def available(self):
        return max(0, self.capital - self._state.get('allocated', 0))

    @property
    def allocated(self):
        return self._state.get('allocated', 0)

    def set_capital(self, amount):
        self._state['capital'] = amount
        self._save()

    def allocate(self, amount):
        self._state['allocated'] = self._state.get('allocated', 0) + amount
        self._save()

    def release(self, amount, pnl=0):
        self._state['allocated'] = max(0, self._state.get('allocated', 0) - amount)
        self._state['capital'] = self._state.get('capital', 0) + pnl
        self._save()

    def calculate_position_size(self, risk_pct=None):
        if risk_pct is None:
            risk_pct = TradingConfig.MAX_RISK_PER_TRADE_PCT
        max_by_risk = self.capital * (risk_pct / 100)
        max_by_exposure = self.capital * (TradingConfig.MAX_EXPOSURE_TOTAL_PCT / 100) - self.allocated
        return max(0, round(min(max_by_risk, max(0, max_by_exposure))))

    def check_scale_eligibility(self, performance):
        eligible = []
        for target, (min_t, min_wr, min_pf) in TradingConfig.SCALE_REQUIREMENTS.items():
            if self.capital >= target:
                continue
            if (performance.get('total_trades', 0) >= min_t and
                performance.get('win_rate', 0) >= min_wr and
                performance.get('profit_factor', 0) >= min_pf):
                eligible.append(target)
        return min(eligible) if eligible else None

    def get_status(self):
        return {
            'capital': self.capital, 'allocated': self.allocated,
            'available': self.available,
            'utilization_pct': round(self.allocated / self.capital * 100, 1) if self.capital > 0 else 0,
        }


class CircuitBreaker:
    """Safety mechanisms to prevent catastrophic losses."""

    def __init__(self):
        self._paused_until = 0
        self._consecutive_losses = 0
        self._pause_reason = None

    def check(self, journal, capital_mgr):
        now = time.time()
        if now < self._paused_until:
            remaining = int(self._paused_until - now)
            return False, f"Cooldown ({remaining}s): {self._pause_reason}"

        today_pnl = journal.get_today_pnl()
        if today_pnl < -TradingConfig.MAX_DAILY_LOSS_IDR:
            self._paused_until = now + 86400
            self._pause_reason = f"Max daily loss: Rp {today_pnl:,.0f}"
            return False, self._pause_reason

        if journal.get_today_trade_count() >= TradingConfig.MAX_DAILY_TRADES:
            return False, f"Max daily trades ({TradingConfig.MAX_DAILY_TRADES})"

        if capital_mgr.capital < capital_mgr._state.get('initial_capital', 50000) * 0.5:
            self._paused_until = now + 86400
            self._pause_reason = "Capital < 50% initial — emergency pause"
            return False, self._pause_reason

        if capital_mgr.available < 10000:
            return False, "Available capital < Rp10,000"

        return True, "OK"

    def record_trade_result(self, is_win):
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3:
                self._paused_until = time.time() + TradingConfig.COOLDOWN_AFTER_3_LOSSES
                self._pause_reason = "3 consecutive losses"
                self._consecutive_losses = 0
            else:
                self._paused_until = time.time() + TradingConfig.COOLDOWN_AFTER_LOSS_SECONDS
                self._pause_reason = "Post-loss cooldown"

    def force_pause(self, seconds, reason="Manual"):
        self._paused_until = time.time() + seconds
        self._pause_reason = reason


class PositionMonitor:
    """Real-time monitoring with dynamic trailing stops."""

    def __init__(self, trade_api, journal):
        self.api = trade_api
        self.journal = journal
        self._positions = {}

    def add_position(self, trade_id, pair, entry_price, amount_idr,
                     amount_coin, sl_price=None, tp_price=None, direction='buy'):
        self._positions[trade_id] = {
            'pair': pair, 'entry_price': entry_price,
            'amount_idr': amount_idr, 'amount_coin': amount_coin,
            'direction': direction, 'highest_price': entry_price,
            'trailing_active': False, 'partial_tp_taken': [],
            'sl_price': sl_price, 'tp_price': tp_price,
        }

    def check_positions(self):
        actions = []
        for trade_id, pos in list(self._positions.items()):
            try:
                price_data = self.api.get_price(pos['pair'])
                if not price_data.get('success'):
                    continue
                current = price_data.get('last', 0)
                if current <= 0:
                    continue

                entry = pos['entry_price']
                change_pct = ((current - entry) / entry) * 100

                if current > pos['highest_price']:
                    pos['highest_price'] = current

                # --- Dynamic SL (ATR-based or fixed) ---
                if pos.get('sl_price') and current <= pos['sl_price']:
                    actions.append({'action': 'close', 'trade_id': trade_id,
                                    'reason': 'stop_loss', 'price': current, 'change_pct': change_pct})
                    continue
                elif not pos.get('sl_price') and change_pct <= -TradingConfig.DEFAULT_SL_PCT:
                    actions.append({'action': 'close', 'trade_id': trade_id,
                                    'reason': 'stop_loss', 'price': current, 'change_pct': change_pct})
                    continue

                # --- TP hit ---
                if pos.get('tp_price') and current >= pos['tp_price']:
                    actions.append({'action': 'close', 'trade_id': trade_id,
                                    'reason': 'take_profit', 'price': current, 'change_pct': change_pct})
                    continue

                # --- Trailing Stop ---
                if change_pct >= TradingConfig.TRAILING_STOP_ACTIVATION_PCT:
                    pos['trailing_active'] = True
                if pos['trailing_active']:
                    drop = ((pos['highest_price'] - current) / pos['highest_price']) * 100
                    if drop >= TradingConfig.TRAILING_STOP_DISTANCE_PCT:
                        actions.append({'action': 'close', 'trade_id': trade_id,
                                        'reason': 'trailing_stop', 'price': current,
                                        'change_pct': change_pct, 'high': pos['highest_price']})
                        continue

                # --- Partial Take-Profit ---
                for tp_pct, tp_frac in TradingConfig.PARTIAL_TP_LEVELS:
                    if tp_pct not in pos['partial_tp_taken'] and change_pct >= tp_pct:
                        actions.append({'action': 'partial_close', 'trade_id': trade_id,
                                        'reason': f'partial_tp_{tp_pct}%', 'price': current,
                                        'fraction': tp_frac, 'change_pct': change_pct})
                        pos['partial_tp_taken'].append(tp_pct)

            except Exception as e:
                print(f"[Monitor] Error {trade_id[:8]}: {e}")
        return actions

    def remove_position(self, trade_id):
        self._positions.pop(trade_id, None)

    def get_positions_summary(self):
        summaries = []
        for tid, pos in self._positions.items():
            try:
                pd = self.api.get_price(pos['pair'])
                cur = pd.get('last', 0) if pd.get('success') else 0
                chg = ((cur - pos['entry_price']) / pos['entry_price'] * 100) if cur > 0 else 0
                summaries.append({
                    'trade_id': tid[:8], 'pair': pos['pair'],
                    'entry': pos['entry_price'], 'current': cur,
                    'change_pct': round(chg, 2), 'amount_idr': pos['amount_idr'],
                    'trailing': pos['trailing_active'],
                    'sl': pos.get('sl_price'), 'tp': pos.get('tp_price'),
                })
            except Exception:
                pass
        return summaries


class SignalAnalyzer:
    """
    Deep signal analysis integrating ALL existing modules.
    This is the brain that decides whether to trade.
    """

    def __init__(self, api):
        self.api = api
        # Import existing analysis modules
        try:
            from backend.signal_precision import ProbabilisticSignal, SignalQualityGate, RegimeSwitcher
            self.prob_signal = ProbabilisticSignal()
            self.quality_gate = SignalQualityGate()
            self.regime_switcher = RegimeSwitcher()
            self._has_signal = True
        except Exception as e:
            print(f"[Signal] signal_precision import failed: {e}")
            self._has_signal = False

        try:
            from backend.multi_tf_engine import MultiTimeframeEngine
            self.mtf = MultiTimeframeEngine()
            self._has_mtf = True
        except Exception as e:
            print(f"[Signal] multi_tf import failed: {e}")
            self._has_mtf = False

        try:
            from backend.adaptive_engine import ManipulationDetector, EmotionFilter
            self.manipulation = ManipulationDetector()
            self.emotion_filter = EmotionFilter()
            self._has_adaptive = True
        except Exception as e:
            print(f"[Signal] adaptive_engine import failed: {e}")
            self._has_adaptive = False

        try:
            from backend.market_scanner import MarketRegimeDetector
            self.market_regime = MarketRegimeDetector()
            self._has_regime = True
        except Exception as e:
            print(f"[Signal] market_regime import failed: {e}")
            self._has_regime = False

        try:
            from backend.news_sentiment import NewsSentimentEngine
            self.sentiment = NewsSentimentEngine()
            self._has_sentiment = True
        except Exception as e:
            print(f"[Signal] sentiment engine import failed: {e}")
            self._has_sentiment = False

    def analyze_coin(self, pair, dynamic_weights=None):
        """
        Deep analysis of a single coin using all modules (ML-weighted).
        Returns: {should_trade, confidence, direction, reasons, sl_price, tp_price}
        """
        dynamic_weights = dynamic_weights or {}
        result = {
            'should_trade': False, 'confidence': 0, 'direction': 'none',
            'reasons': [], 'sl_price': None, 'tp_price': None,
            'regime': 'UNKNOWN', 'factors': {},
        }

        try:
            # 1. Get kline data (15m candles for analysis)
            candles = self.api.get_kline(pair, resolution='15')
            if not candles or len(candles) < 30:
                result['reasons'].append("Insufficient candle data")
                return result

            closes = [c['close'] for c in candles]
            highs = [c['high'] for c in candles]
            lows = [c['low'] for c in candles]
            volumes = [c['vol'] for c in candles]
            current_price = closes[-1]

            # 2. Get order book data for whale/pressure analysis
            whale_ratio = 0.5
            buy_pressure = 0.5
            wall_imbalance = 0
            try:
                depth = self.api.get_depth(pair)
                if depth:
                    buy_orders = depth.get('buy', [])
                    sell_orders = depth.get('sell', [])
                    if buy_orders and sell_orders:
                        buy_vol = sum(float(o[1]) for o in buy_orders[:20])
                        sell_vol = sum(float(o[1]) for o in sell_orders[:20])
                        total = buy_vol + sell_vol
                        if total > 0:
                            buy_pressure = buy_vol / total
                            whale_ratio = buy_pressure
                            wall_imbalance = (buy_vol - sell_vol) / total
            except Exception:
                pass

            # 3. Probabilistic Signal (5-factor)
            if self._has_signal:
                try:
                    sig = self.prob_signal.compute(
                        closes=closes, highs=highs, lows=lows, volumes=volumes,
                        whale_ratio=whale_ratio, buy_pressure=buy_pressure,
                        wall_imbalance=wall_imbalance,
                    )
                    result['confidence'] = sig.get('confidence', 0)
                    result['factors'] = sig.get('factors', {})

                    bullish = sig.get('bullish_pct', 0)
                    bearish = sig.get('bearish_pct', 0)

                    if bullish > bearish and bullish > 50:
                        result['direction'] = 'bullish'
                    elif bearish > bullish and bearish > 50:
                        result['direction'] = 'bearish'
                    else:
                        result['direction'] = 'sideways'
                        result['reasons'].append("No clear direction")
                except Exception as e:
                    result['reasons'].append(f"Signal error: {e}")

            # 4. Market Regime Detection
            regime = 'TRENDING'
            if self._has_regime:
                try:
                    reg = self.market_regime.detect(candles, lookback=30)
                    regime = reg.get('regime', 'TRENDING')
                    result['regime'] = regime

                    # Adjust confidence for regime
                    if result['direction'] == 'bullish':
                        result['confidence'] = self.market_regime.adjust_confidence(
                            result['confidence'], regime, 'bullish'
                        )
                except Exception:
                    pass

            # 5. Quality Gate
            if self._has_signal:
                try:
                    # Get spread
                    price_data = self.api.get_price(pair)
                    spread_pct = 0
                    if price_data.get('success'):
                        buy_p = price_data.get('buy', 0)
                        sell_p = price_data.get('sell', 0)
                        if buy_p > 0:
                            spread_pct = ((sell_p - buy_p) / buy_p) * 100

                    # Calculate ATR for spread filter
                    atr = 0
                    if len(highs) >= 14:
                        trs = []
                        for i in range(1, min(14, len(highs))):
                            tr = max(highs[i] - lows[i],
                                     abs(highs[i] - closes[i-1]),
                                     abs(lows[i] - closes[i-1]))
                            trs.append(tr)
                        atr = sum(trs) / len(trs) if trs else 0

                    gate = self.quality_gate.check(
                        signal={'confidence': result['confidence'],
                                'bullish_pct': 60 if result['direction'] == 'bullish' else 40},
                        regime=regime,
                        spread_pct=spread_pct,
                        atr=atr,
                    )
                    if not gate.get('passed', False):
                        result['reasons'].append(f"Quality gate: {gate.get('reasons', [])}")
                        result['should_trade'] = False
                        return result
                except Exception as e:
                    result['reasons'].append(f"Gate error: {e}")

            # 6. Manipulation Check
            if self._has_adaptive:
                try:
                    manip = self.manipulation.full_scan(closes, volumes)
                    risk = manip.get('overall_risk', 'LOW')
                    if risk in ['HIGH', 'CRITICAL']:
                        result['reasons'].append(f"Manipulation risk: {risk}")
                        return result

                    # Emotion filter (choppy market)
                    if len(highs) >= 20:
                        atr_vals = []
                        for i in range(1, 15):
                            tr = max(highs[-i] - lows[-i],
                                     abs(highs[-i] - closes[-i-1]),
                                     abs(lows[-i] - closes[-i-1]))
                            atr_vals.append(tr)
                        bb_width = (max(closes[-20:]) - min(closes[-20:])) / current_price * 100
                        rsi_val = self._quick_rsi(closes)

                        if self.emotion_filter.should_filter(
                            volatility_regime=regime,
                            rsi=rsi_val,
                            macd_hist=0,
                            bb_width=bb_width,
                        ):
                            result['reasons'].append("Emotion filter: too choppy")
                            return result
                except Exception:
                    pass

            # 7. Multi-Timeframe Confirmation (bonus confidence)
            if self._has_mtf:
                try:
                    candles_1h = self.api.get_kline(pair, resolution='60')
                    if candles_1h and len(candles_1h) >= 20:
                        tf_15m = self.mtf.analyze_single_tf(candles, '15m')
                        tf_1h = self.mtf.analyze_single_tf(candles_1h, '1h')

                        # If both timeframes agree on direction, boost confidence
                        tf15_bullish = tf_15m.get('score', 0) > 0
                        tf1h_bullish = tf_1h.get('score', 0) > 0

                        if tf15_bullish and tf1h_bullish and result['direction'] == 'bullish':
                            result['confidence'] += TradingConfig.CONFIDENCE_BOOST_MULTI_TF
                            result['reasons'].append("Multi-TF aligned: +5 confidence")
                        elif not tf15_bullish or not tf1h_bullish:
                            result['confidence'] -= 5
                            result['reasons'].append("Multi-TF divergent: -5 confidence")
                except Exception:
                    pass

            # 8. Sentiment Analysis Integration
            if self._has_sentiment:
                try:
                    sent = self.sentiment.analyze_coin_sentiment(pair.replace('idr', ''))
                    sent_score = sent.get('score', 0)
                    if sent_score > 0.5:
                        result['confidence'] += 3
                        result['reasons'].append("Bullish sentiment: +3 conf")
                        result['factors']['sentiment'] = 1.0
                    elif sent_score < -0.5:
                        result['confidence'] -= 5
                        result['reasons'].append("Bearish sentiment: -5 conf")
                        result['factors']['sentiment'] = -1.0
                except Exception:
                    pass

            # 9. Apply Continuous Learning (Dynamic Weights)
            if dynamic_weights:
                adjustment = 0
                for factor, weight in dynamic_weights.items():
                    if factor in result['factors'] and result['factors'][factor] > 0:
                        adjustment += weight
                
                if adjustment != 0:
                    result['confidence'] += adjustment
                    result['reasons'].append(f"AI Learning Adjust: {adjustment:+.1f}")

            # 10. Calculate dynamic SL/TP using ATR
            if atr > 0:
                result['sl_price'] = round(current_price - (atr * TradingConfig.ATR_SL_MULTIPLIER), 2)
                result['tp_price'] = round(current_price + (atr * TradingConfig.ATR_TP_MULTIPLIER), 2)

            # Final decision
            if (result['direction'] == 'bullish' and
                result['confidence'] >= TradingConfig.MIN_CONFIDENCE_TO_TRADE):
                result['should_trade'] = True

        except Exception as e:
            result['reasons'].append(f"Analysis error: {e}")
            print(f"[Signal] analyze_coin error: {traceback.format_exc()}")

        return result

    def _quick_rsi(self, closes, period=14):
        if len(closes) < period + 1:
            return 50
        gains, losses = [], []
        for i in range(-period, 0):
            d = closes[i] - closes[i-1]
            gains.append(max(0, d))
            losses.append(max(0, -d))
        avg_g = sum(gains) / period
        avg_l = sum(losses) / period
        if avg_l == 0:
            return 100
        rs = avg_g / avg_l
        return round(100 - (100 / (1 + rs)), 1)


class PostTradeAnalyzer:
    """Learn from every trade to improve future decisions."""

    def __init__(self, data_dir='data'):
        os.makedirs(data_dir, exist_ok=True)
        self._filepath = os.path.join(data_dir, 'learning_state.json')
        self._state = self._load()

    def _load(self):
        try:
            if os.path.exists(self._filepath):
                with open(self._filepath, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            'factor_adjustments': {},  # factor_name -> running adjustment
            'mistake_log': [],
            'confidence_threshold_offset': 0,  # Dynamic threshold adjustment
            'total_analyzed': 0,
        }

    def _save(self):
        try:
            with open(self._filepath, 'w') as f:
                json.dump(self._state, f, indent=2, default=str)
        except Exception:
            pass

    def analyze_trade(self, trade):
        """Analyze a closed trade and adjust weights."""
        if trade.get('status') != 'closed':
            return

        pnl = trade.get('pnl_idr', 0) or 0
        factors = trade.get('signal_factors', {})
        confidence = trade.get('confidence', 0)

        self._state['total_analyzed'] = self._state.get('total_analyzed', 0) + 1

        # Classify the trade outcome
        if pnl > 0:
            category = 'profitable'
        elif pnl > -500:
            category = 'small_loss'
        else:
            category = 'significant_loss'

        # Record mistake if it was a loss
        if pnl < 0:
            mistake = {
                'time': datetime.utcnow().isoformat(),
                'pair': trade.get('pair'),
                'pnl_idr': pnl,
                'confidence_at_entry': confidence,
                'exit_reason': trade.get('exit_reason'),
                'category': self._categorize_mistake(trade),
            }
            self._state['mistake_log'].append(mistake)
            # Keep only last 50 mistakes
            self._state['mistake_log'] = self._state['mistake_log'][-50:]

            # If entry confidence was below 70, raise the threshold
            if confidence < 70:
                offset = self._state.get('confidence_threshold_offset', 0)
                self._state['confidence_threshold_offset'] = min(10, offset + 1)

        elif pnl > 0 and confidence > 75:
            # Good trade with high confidence — slightly lower threshold
            offset = self._state.get('confidence_threshold_offset', 0)
            self._state['confidence_threshold_offset'] = max(-5, offset - 0.5)

        # Adjust factor weights based on outcome
        for factor_name, factor_score in factors.items():
            adj = self._state.get('factor_adjustments', {})
            current = adj.get(factor_name, 0)
            if pnl > 0:
                adj[factor_name] = current + 0.5  # Factor was useful
            else:
                adj[factor_name] = current - 0.3  # Factor misled us
            self._state['factor_adjustments'] = adj

        self._save()

    def _categorize_mistake(self, trade):
        reason = trade.get('exit_reason', '')
        if reason == 'stop_loss':
            return 'bad_entry'
        elif reason == 'trailing_stop':
            return 'gave_back_profit'
        elif reason == 'emergency_close':
            return 'market_crash'
        else:
            return 'unknown'

    def get_adjusted_threshold(self):
        """Get the dynamically adjusted confidence threshold."""
        base = TradingConfig.MIN_CONFIDENCE_TO_TRADE
        offset = self._state.get('confidence_threshold_offset', 0)
        return base + offset

    def get_factor_adjustments(self):
        """Get ML-tuned weight adjustments for factors."""
        return self._state.get('factor_adjustments', {})

    def get_learning_summary(self):
        total = self._state.get('total_analyzed', 0)
        mistakes = self._state.get('mistake_log', [])
        offset = self._state.get('confidence_threshold_offset', 0)

        if not mistakes:
            top_mistake = "N/A"
        else:
            cats = {}
            for m in mistakes:
                c = m.get('category', 'unknown')
                cats[c] = cats.get(c, 0) + 1
            top_mistake = max(cats, key=cats.get) if cats else "N/A"

        return {
            'trades_analyzed': total,
            'top_mistake_type': top_mistake,
            'threshold_offset': offset,
            'effective_threshold': TradingConfig.MIN_CONFIDENCE_TO_TRADE + offset,
        }


class TradingEngine:
    """
    Main autonomous trading engine V2.
    Orchestrates: Scan → Deep Analysis → Gate → Risk → Execute → Monitor → Learn
    """

    def __init__(self, trade_api, analysis_api=None, notifier=None):
        from backend.trade_journal import TradeJournal
        from backend.coin_scanner import CoinScanner

        self.trade_api = trade_api
        self.analysis_api = analysis_api or trade_api
        self.notify = notifier or (lambda x: print(f"[Notify] {x}"))

        # Core components
        self.journal = TradeJournal()
        self.scanner = CoinScanner(self.analysis_api)
        self.capital = CapitalManager(initial_capital=TradingConfig.INITIAL_CAPITAL)
        self.circuit_breaker = CircuitBreaker()
        self.monitor = PositionMonitor(self.trade_api, self.journal)
        self.signal_analyzer = SignalAnalyzer(self.analysis_api)
        self.learner = PostTradeAnalyzer()

        # State
        self._enabled = False
        self._running = False
        self._thread = None
        self._last_scan_time = 0
        self._last_position_check = 0
        self._last_daily_report = 0
        self._error_count = 0

    # ==========================================
    # CONTROL
    # ==========================================

    def enable(self):
        self._enabled = True
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._main_loop, daemon=True)
            self._thread.start()
            threshold = self.learner.get_adjusted_threshold()
            self.notify(
                "🟢 *Auto-Trading AKTIF (V2 AI)*\n"
                "━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💼 Modal: Rp {self.capital.capital:,.0f}\n"
                f"🎯 Min Confidence: {threshold}%\n"
                f"🛡️ Max Risk/Trade: {TradingConfig.MAX_RISK_PER_TRADE_PCT}%\n"
                f"📊 Max Positions: {TradingConfig.MAX_OPEN_POSITIONS}\n"
                f"⏰ Scan: setiap {TradingConfig.SCAN_INTERVAL_SECONDS}s\n"
                f"🔍 Posisi check: setiap {TradingConfig.POSITION_CHECK_SECONDS}s"
            )
        return True

    def disable(self):
        self._enabled = False
        self._running = False
        self.notify("🔴 *Auto-Trading NONAKTIF*\nPosisi terbuka tetap dimonitor.")
        return True

    def emergency_close_all(self):
        open_trades = self.journal.get_open_trades()
        closed = 0
        for trade in open_trades:
            try:
                pd = self.trade_api.get_price(trade.get('pair', ''))
                if pd.get('success'):
                    self._execute_close(trade['id'], pd.get('last', 0), 'emergency_close')
                    closed += 1
            except Exception as e:
                print(f"[Emergency] {e}")
        self.circuit_breaker.force_pause(3600, "Emergency close")
        self.notify(f"🚨 *EMERGENCY CLOSE* — {closed} posisi ditutup.")
        return closed

    # ==========================================
    # MAIN AUTONOMOUS LOOP (V2 — Robust)
    # ==========================================

    def _main_loop(self):
        print("[Engine V2] ▶ Autonomous trading loop started")
        self._error_count = 0

        while self._running:
            try:
                now = time.time()

                # Position monitoring (every 10s)
                if now - self._last_position_check >= TradingConfig.POSITION_CHECK_SECONDS:
                    self._check_positions()
                    self._last_position_check = now

                # Market scan + trade decisions (every 60s)
                if self._enabled and (now - self._last_scan_time >= TradingConfig.SCAN_INTERVAL_SECONDS):
                    self._scan_and_trade()
                    self._last_scan_time = now

                # Daily auto-report (midnight WIB)
                if now - self._last_daily_report >= 86400:
                    utc_hour = datetime.utcnow().hour
                    if utc_hour == TradingConfig.DAILY_REPORT_HOUR_UTC:
                        self._send_daily_report()
                        self._last_daily_report = now

                self._error_count = max(0, self._error_count - 1)  # Decay errors
                time.sleep(3)

            except Exception as e:
                self._error_count += 1
                print(f"[Engine V2] Loop error #{self._error_count}: {e}")

                # Exponential backoff on repeated errors
                if self._error_count >= 10:
                    print("[Engine V2] Too many errors — pausing 5 min")
                    self.notify("⚠️ Trading engine: terlalu banyak error, pause 5 menit.")
                    time.sleep(300)
                    self._error_count = 0
                else:
                    time.sleep(min(30, 5 * self._error_count))

        print("[Engine V2] ⏹ Trading loop stopped")

    def _scan_and_trade(self):
        """
        Enhanced scan pipeline:
        1. Circuit breaker → 2. Scan market → 3. Deep signal analysis
        → 4. Confidence check → 5. Position sizing → 6. Execute
        """
        # Circuit breaker check
        is_safe, reason = self.circuit_breaker.check(self.journal, self.capital)
        if not is_safe:
            return

        # Check max open positions
        open_count = len(self.journal.get_open_trades())
        if open_count >= TradingConfig.MAX_OPEN_POSITIONS:
            return

        # Scan for candidates
        top_coins = self.scanner.get_top_coins(5)

        for coin in top_coins:
            if coin['score'] < TradingConfig.MIN_SCAN_SCORE:
                continue
            if coin['volume_idr'] < TradingConfig.MIN_VOLUME_IDR:
                continue
            if coin['spread_pct'] > TradingConfig.MAX_SPREAD_PCT:
                continue

            # No duplicate positions
            if any(t.get('pair') == coin['pair'] for t in self.journal.get_open_trades()):
                continue

            # === DEEP SIGNAL ANALYSIS (the V2 brain with AI learning) ===
            ml_weights = self.learner.get_factor_adjustments()
            analysis = self.signal_analyzer.analyze_coin(coin['pair'], dynamic_weights=ml_weights)

            if not analysis['should_trade']:
                continue

            # Dynamic confidence threshold (adjusted by learner)
            threshold = self.learner.get_adjusted_threshold()
            if analysis['confidence'] < threshold:
                continue

            # Slippage protection: check if bid/ask is reasonable vs last price
            try:
                pd = self.trade_api.get_price(coin['pair'])
                if pd.get('success'):
                    last = pd.get('last', 0)
                    ask = pd.get('sell', 0) or coin['ask']
                    if last > 0 and ask > 0:
                        slippage = abs(ask - last) / last * 100
                        if slippage > TradingConfig.MAX_SLIPPAGE_PCT:
                            continue
            except Exception:
                pass

            # Position sizing
            position_idr = self.capital.calculate_position_size()
            if position_idr < 15000:
                continue

            # Re-check circuit breaker
            is_safe, _ = self.circuit_breaker.check(self.journal, self.capital)
            if not is_safe:
                break

            # Execute buy with deep analysis results
            self._execute_buy(
                pair=coin['pair'],
                price=coin['ask'],
                amount_idr=position_idr,
                confidence=analysis['confidence'],
                factors=analysis['factors'],
                sl_price=analysis.get('sl_price'),
                tp_price=analysis.get('tp_price'),
                regime=analysis.get('regime', 'UNKNOWN'),
            )

            break  # One trade per cycle

    def _execute_buy(self, pair, price, amount_idr, confidence, factors,
                     sl_price=None, tp_price=None, regime='UNKNOWN'):
        trade_id = str(uuid.uuid4())
        try:
            result = self.trade_api.create_order(
                pair=pair, order_type='buy', price=price, amount_idr=amount_idr,
            )
            if not result.get('success'):
                print(f"[Trade] Buy failed {pair}: {result.get('error')}")
                return False

            amount_coin = amount_idr / price if price > 0 else 0

            self.journal.record_open(
                trade_id=trade_id, pair=pair, direction='buy',
                entry_price=price, amount_idr=amount_idr,
                amount_coin=amount_coin, confidence=confidence,
                signal_factors=factors, order_id=result.get('order_id'),
            )
            self.capital.allocate(amount_idr)
            self.monitor.add_position(
                trade_id=trade_id, pair=pair, entry_price=price,
                amount_idr=amount_idr, amount_coin=amount_coin,
                sl_price=sl_price, tp_price=tp_price,
            )

            coin_name = pair.replace('idr', '').upper()
            sl_str = f"Rp {sl_price:,.0f}" if sl_price else f"-{TradingConfig.DEFAULT_SL_PCT}%"
            tp_str = f"Rp {tp_price:,.0f}" if tp_price else f"+{TradingConfig.DEFAULT_TP_PCT}%"

            self.notify(
                f"🟢 *BELI {coin_name}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Harga: Rp {price:,.0f}\n"
                f"📊 Amount: Rp {amount_idr:,.0f}\n"
                f"🎯 Confidence: {confidence}%\n"
                f"🌍 Regime: {regime}\n"
                f"🛡️ SL: {sl_str} | TP: {tp_str}\n"
                f"📎 ID: {trade_id[:8]}"
            )
            return True
        except Exception as e:
            print(f"[Trade] BUY error: {e}")
            return False

    def _execute_close(self, trade_id, current_price, reason='manual'):
        trade = None
        for t in self.journal.get_open_trades():
            if t['id'] == trade_id:
                trade = t
                break
        if not trade:
            return False

        pair = trade['pair']
        coin_name = pair.replace('idr', '').upper()

        try:
            self.trade_api.create_order(
                pair=pair, order_type='sell', price=current_price,
                amount_coin=trade.get('amount_coin', 0),
            )
            closed = self.journal.record_close(trade_id, current_price, reason)
            if closed:
                pnl = closed.get('pnl_idr', 0)
                pnl_pct = closed.get('pnl_pct', 0)

                self.capital.release(trade['amount_idr'], pnl)
                self.monitor.remove_position(trade_id)
                self.circuit_breaker.record_trade_result(pnl > 0)

                # Self-learning: analyze this trade
                self.learner.analyze_trade(closed)

                emoji = "🟢" if pnl > 0 else "🔴"
                self.notify(
                    f"{emoji} *JUAL {coin_name}*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 Entry: Rp {trade['entry_price']:,.0f} → Exit: Rp {current_price:,.0f}\n"
                    f"📊 P&L: Rp {pnl:,.0f} ({pnl_pct:+.2f}%)\n"
                    f"📝 Alasan: {reason}\n"
                    f"💼 Modal: Rp {self.capital.capital:,.0f}"
                )
            return True
        except Exception as e:
            print(f"[Trade] SELL error: {e}")
            return False

    def _check_positions(self):
        for action in self.monitor.check_positions():
            if action['action'] in ['close']:
                self._execute_close(action['trade_id'], action['price'], action['reason'])
            elif action['action'] == 'partial_close':
                print(f"[Monitor] Partial TP: {action['reason']}")

    def _send_daily_report(self):
        perf = self.journal.get_performance()
        today = self.journal.get_today_stats()
        learn = self.learner.get_learning_summary()
        scale = self.capital.check_scale_eligibility(perf)

        lines = [
            "📊 *Laporan Harian Trading AI*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"📅 {datetime.utcnow().strftime('%Y-%m-%d')}",
            f"💼 Modal: Rp {self.capital.capital:,.0f}",
            f"📈 Hari ini: {today.get('trades', 0)} trades | Rp {today.get('total_pnl', 0):,.0f}",
            f"📊 30-hari: WR {perf['win_rate']}% | PF {perf['profit_factor']}",
            f"🧠 AI Threshold: {learn['effective_threshold']}% (offset: {learn['threshold_offset']:+.0f})",
            f"🔍 Top mistake: {learn['top_mistake_type']}",
        ]
        if scale:
            lines.append(f"⬆️ Scale eligible: Rp {scale:,.0f}")

        self.notify("\n".join(lines))

    # ==========================================
    # STATUS & REPORTING
    # ==========================================

    def get_status(self):
        cap = self.capital.get_status()
        perf = self.journal.get_performance()
        today = self.journal.get_today_stats()
        positions = self.monitor.get_positions_summary()
        scale_to = self.capital.check_scale_eligibility(perf)
        learn = self.learner.get_learning_summary()
        return {
            'enabled': self._enabled, 'capital': cap, 'performance': perf,
            'today': today, 'open_positions': positions,
            'scale_eligible': scale_to, 'learning': learn,
        }

    def format_status(self):
        s = self.get_status()
        cap = s['capital']
        perf = s['performance']
        today = s['today']
        learn = s.get('learning', {})
        emoji = "🟢" if self._enabled else "🔴"

        lines = [
            f"{emoji} *Trading Engine V2 AI*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"💼 Modal: Rp {cap['capital']:,.0f} (terpakai: {cap['utilization_pct']}%)",
            f"💰 Tersedia: Rp {cap['available']:,.0f}",
            "",
            f"📈 *Hari Ini:* {today.get('trades', 0)} trades | Rp {today.get('total_pnl', 0):,.0f}",
            f"📊 *30 Hari:* WR {perf.get('win_rate', 0)}% | PF {perf.get('profit_factor', 0)}",
            f"   Total P&L: Rp {perf.get('total_pnl', 0):,.0f}",
            f"🧠 *AI:* Threshold {learn.get('effective_threshold', 65)}%",
        ]

        positions = s.get('open_positions', [])
        if positions:
            lines.append(f"\n📋 *Posisi ({len(positions)})*")
            for p in positions:
                e = "🟢" if p['change_pct'] > 0 else "🔴"
                t = " 🔄" if p['trailing'] else ""
                lines.append(f"   {e} {p['pair'].replace('idr','').upper()} {p['change_pct']:+.2f}%{t}")

        if s.get('scale_eligible'):
            lines.append(f"\n⬆️ Scale to: Rp {s['scale_eligible']:,.0f}")

        return "\n".join(lines)

    def format_performance(self):
        perf = self.journal.get_performance()
        recent = self.journal.get_recent_trades(5)
        learn = self.learner.get_learning_summary()

        lines = [
            "📊 *Performance Report*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"Trades: {perf['total_trades']} ({perf.get('wins',0)}W/{perf.get('losses',0)}L)",
            f"Win Rate: {perf['win_rate']}% | PF: {perf['profit_factor']}",
            f"Avg Win: Rp {perf['avg_win']:,.0f} | Avg Loss: Rp {perf['avg_loss']:,.0f}",
            f"Best: Rp {perf['best_trade']:,.0f} | Worst: Rp {perf['worst_trade']:,.0f}",
            f"Drawdown: Rp {perf['max_drawdown']:,.0f}",
            f"Total P&L: Rp {perf['total_pnl']:,.0f}",
            "",
            f"🧠 *AI Learning*",
            f"   Analyzed: {learn['trades_analyzed']} trades",
            f"   Threshold: {learn['effective_threshold']}%",
            f"   Top mistake: {learn['top_mistake_type']}",
        ]

        if recent:
            lines.append("\n📋 *Recent*")
            for t in recent:
                e = "🟢" if (t.get('pnl_idr') or 0) > 0 else ("🔴" if t['status'] == 'closed' else "⏳")
                c = t.get('pair', '?').replace('idr', '').upper()
                p = f"Rp {t['pnl_idr']:,.0f}" if t.get('pnl_idr') is not None else "Open"
                lines.append(f"   {e} {c} | {p}")

        return "\n".join(lines)
