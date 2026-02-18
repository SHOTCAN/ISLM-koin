"""
ISLM Monitor — Adaptive Learning & Manipulation Detection Engine V8
====================================================================
Institutional-grade adaptive system:
  - Signal history logging + auto evaluation (1H/4H/24H)
  - Win rate tracking + dynamic weight recalibration (30-90d)
  - Manipulation detector (wash trading, pump dump, Z-score spike)
  - Drawdown control system (pause signals if winrate <50%)
  - Emotion filter mode during high chop/volatility
  - Model versioning log (V8→V9 tracking)
  - AI weekly self-diagnostic performance report
"""

import time
import json
import os
import statistics
from datetime import datetime, timedelta
from collections import deque


class SignalTracker:
    """Track all signals and evaluate performance over time."""

    def __init__(self, data_dir='data'):
        self._signals = []
        self._data_dir = data_dir
        self._history_file = os.path.join(data_dir, 'signal_history.json')
        self._load_history()

    def _load_history(self):
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            if os.path.exists(self._history_file):
                with open(self._history_file, 'r') as f:
                    self._signals = json.load(f)
        except:
            self._signals = []

    def _save_history(self):
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            with open(self._history_file, 'w') as f:
                json.dump(self._signals[-500:], f, indent=2)  # Keep last 500
        except:
            pass

    def record_signal(self, signal_type, direction, price, confidence, factors=None):
        """Record a new signal with timestamp and entry price."""
        signal = {
            'id': len(self._signals) + 1,
            'timestamp': time.time(),
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'type': signal_type,
            'direction': direction,  # BUY / SELL / HOLD
            'entry_price': price,
            'confidence': confidence,
            'factors': factors or {},
            'evaluations': {},  # Will be filled by evaluate()
            'status': 'OPEN',
        }
        self._signals.append(signal)
        self._save_history()
        return signal['id']

    def evaluate_signals(self, current_price):
        """Auto-evaluate all open signals at 1H, 4H, 24H marks."""
        now = time.time()
        eval_windows = {
            '1H': 3600,
            '4H': 14400,
            '24H': 86400,
        }

        evaluated = 0
        for signal in self._signals:
            if signal['status'] == 'CLOSED':
                continue

            age = now - signal['timestamp']
            entry = signal['entry_price']
            if entry <= 0:
                continue

            for window_name, window_secs in eval_windows.items():
                if window_name in signal['evaluations']:
                    continue
                if age >= window_secs:
                    pnl_pct = (current_price - entry) / entry * 100
                    if signal['direction'] == 'SELL':
                        pnl_pct = -pnl_pct  # Invert for sell signals

                    signal['evaluations'][window_name] = {
                        'price': current_price,
                        'pnl_pct': round(pnl_pct, 2),
                        'result': 'WIN' if pnl_pct > 0 else 'LOSS',
                        'eval_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    }
                    evaluated += 1

            # Close signal after 24H evaluation
            if '24H' in signal['evaluations']:
                signal['status'] = 'CLOSED'

        if evaluated > 0:
            self._save_history()
        return evaluated

    def get_winrate(self, window='24H', days=30):
        """Calculate win rate for given evaluation window and lookback period."""
        cutoff = time.time() - days * 86400
        relevant = [s for s in self._signals
                     if s['timestamp'] > cutoff
                     and window in s.get('evaluations', {})]

        if not relevant:
            return {'valid': False, 'winrate': 0, 'total': 0}

        wins = sum(1 for s in relevant if s['evaluations'][window]['result'] == 'WIN')
        total = len(relevant)
        losses = total - wins

        pnls = [s['evaluations'][window]['pnl_pct'] for s in relevant]
        avg_pnl = statistics.mean(pnls) if pnls else 0
        avg_win = statistics.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = statistics.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0

        return {
            'valid': True,
            'winrate': round(wins / max(total, 1) * 100, 1),
            'wins': wins,
            'losses': losses,
            'total': total,
            'avg_pnl': round(avg_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'best': round(max(pnls), 2) if pnls else 0,
            'worst': round(min(pnls), 2) if pnls else 0,
            'profit_factor': round(abs(avg_win / min(avg_loss, -0.01)), 2) if avg_loss < 0 else 999,
            'window': window,
            'period_days': days,
        }

    def get_performance_report(self):
        """Full performance report across all windows."""
        return {
            '1H': self.get_winrate('1H', 30),
            '4H': self.get_winrate('4H', 30),
            '24H': self.get_winrate('24H', 30),
            '24H_90d': self.get_winrate('24H', 90),
            'total_signals': len(self._signals),
            'open_signals': sum(1 for s in self._signals if s['status'] == 'OPEN'),
        }

    def should_pause_signals(self):
        """Drawdown control: pause if recent winrate <50%."""
        wr = self.get_winrate('24H', 14)  # Last 2 weeks
        if wr['valid'] and wr['total'] >= 5 and wr['winrate'] < 50:
            return True, f"Winrate {wr['winrate']}% (last 14d, {wr['total']} signals)"
        return False, "OK"


class ManipulationDetector:
    """Detect market manipulation patterns."""

    @staticmethod
    def detect_volume_spike(volumes, threshold_z=3.0):
        """Detect abnormal volume spikes using Z-score."""
        if not volumes or len(volumes) < 20:
            return {'detected': False}

        recent = list(volumes)
        mean_vol = statistics.mean(recent[:-1])
        std_vol = statistics.stdev(recent[:-1]) if len(recent) > 2 else 1
        if std_vol == 0:
            std_vol = 1

        current_vol = recent[-1]
        z_score = (current_vol - mean_vol) / std_vol

        if z_score > threshold_z:
            return {
                'detected': True,
                'type': 'VOLUME_SPIKE',
                'z_score': round(z_score, 2),
                'current_vol': current_vol,
                'avg_vol': round(mean_vol, 2),
                'ratio': round(current_vol / max(mean_vol, 1), 1),
                'severity': 'HIGH' if z_score > 5 else 'MEDIUM',
                'alert': f"⚠️ Volume {round(current_vol/max(mean_vol,1), 1)}x rata-rata (Z={z_score:.1f})",
            }
        return {'detected': False, 'z_score': round(z_score, 2)}

    @staticmethod
    def detect_wash_trading(trades):
        """Detect wash trading: rapid buy-sell at same price."""
        if not trades or len(trades) < 10:
            return {'detected': False}

        # Check for same-price same-volume trades in rapid succession
        suspicious = 0
        for i in range(1, len(trades)):
            t1 = trades[i - 1]
            t2 = trades[i]
            same_price = abs(t1.get('price', 0) - t2.get('price', 0)) < 0.01
            same_volume = abs(t1.get('amount', 0) - t2.get('amount', 0)) < 0.01
            opposite = t1.get('type') != t2.get('type')

            if same_price and same_volume and opposite:
                suspicious += 1

        ratio = suspicious / max(len(trades) - 1, 1)
        return {
            'detected': ratio > 0.2,
            'suspicious_trades': suspicious,
            'total_trades': len(trades),
            'ratio': round(ratio, 3),
            'severity': 'HIGH' if ratio > 0.4 else ('MEDIUM' if ratio > 0.2 else 'LOW'),
        }

    @staticmethod
    def detect_pump_dump(prices, volumes, window=20):
        """Detect pump-and-dump: sharp price increase with heavy volume then reversal."""
        if not prices or len(prices) < window:
            return {'detected': False}

        recent_prices = list(prices[-window:])
        recent_vols = list(volumes[-window:]) if volumes else [0] * window

        # Find max pump point
        max_idx = recent_prices.index(max(recent_prices))
        if max_idx < 3 or max_idx > window - 3:
            return {'detected': False}

        # Pump phase (first half rising sharply)
        pump_pct = (recent_prices[max_idx] - recent_prices[0]) / max(recent_prices[0], 1) * 100
        # Dump phase (second half dropping)
        dump_pct = (recent_prices[max_idx] - recent_prices[-1]) / max(recent_prices[max_idx], 1) * 100

        # Volume during pump should be abnormally high
        pump_vol = statistics.mean(recent_vols[:max_idx]) if max_idx > 0 else 0
        dump_vol = statistics.mean(recent_vols[max_idx:]) if max_idx < len(recent_vols) else 0

        is_pump_dump = pump_pct > 5 and dump_pct > 3 and pump_vol > statistics.mean(recent_vols) * 1.5

        return {
            'detected': is_pump_dump,
            'pump_pct': round(pump_pct, 1),
            'dump_pct': round(dump_pct, 1),
            'peak_idx': max_idx,
            'severity': 'HIGH' if pump_pct > 10 and dump_pct > 8 else 'MEDIUM',
            'pump_volume_ratio': round(pump_vol / max(statistics.mean(recent_vols), 1), 1) if recent_vols else 0,
        }

    @staticmethod
    def is_healthy_correction(prices, drop_threshold=5, recovery_ratio=0.3):
        """Distinguish healthy correction vs distribution."""
        if not prices or len(prices) < 10:
            return {'type': 'UNKNOWN'}

        recent = list(prices[-20:])
        peak = max(recent[:10])
        trough = min(recent[5:15])
        current = recent[-1]

        drop_pct = (peak - trough) / max(peak, 1) * 100
        recovery_pct = (current - trough) / max(peak - trough, 1)

        if drop_pct < drop_threshold:
            return {'type': 'NOT_CORRECTION', 'drop_pct': round(drop_pct, 1)}

        if recovery_pct > recovery_ratio:
            return {
                'type': 'HEALTHY_CORRECTION',
                'drop_pct': round(drop_pct, 1),
                'recovery': round(recovery_pct * 100, 1),
                'reason': 'Price recovering — buyers stepping in',
            }
        else:
            return {
                'type': 'DISTRIBUTION',
                'drop_pct': round(drop_pct, 1),
                'recovery': round(recovery_pct * 100, 1),
                'reason': 'Weak recovery — potential distribution/selloff',
            }

    @staticmethod
    def full_scan(prices, volumes, trades=None):
        """Run all manipulation checks. Returns comprehensive report."""
        results = {
            'timestamp': datetime.now().strftime('%H:%M'),
            'checks': {},
        }

        # Volume spike
        if volumes:
            results['checks']['volume_spike'] = ManipulationDetector.detect_volume_spike(volumes)

        # Wash trading
        if trades:
            results['checks']['wash_trading'] = ManipulationDetector.detect_wash_trading(trades)

        # Pump & dump
        if prices and volumes:
            results['checks']['pump_dump'] = ManipulationDetector.detect_pump_dump(prices, volumes)

        # Correction type
        if prices:
            results['checks']['correction'] = ManipulationDetector.is_healthy_correction(prices)

        # Overall risk
        alerts = sum(1 for c in results['checks'].values() if c.get('detected'))
        results['manipulation_risk'] = 'HIGH' if alerts >= 2 else ('MEDIUM' if alerts >= 1 else 'LOW')
        results['alert_count'] = alerts

        return results


class EmotionFilter:
    """Filter out noise during high chop/volatile sideways markets."""

    @staticmethod
    def should_filter(volatility_regime, rsi, macd_hist, bb_width, recent_signals=None):
        """Return True if market is too choppy for reliable signals."""
        chop_score = 0

        # Volatile regime + narrow BB = chop
        if volatility_regime == 'RANGING':
            chop_score += 30
        if bb_width and bb_width < 2:
            chop_score += 20

        # RSI near 50 = no direction
        if 40 < rsi < 60:
            chop_score += 15

        # MACD near zero = no momentum
        if abs(macd_hist) < 0.5:
            chop_score += 15

        # Too many signals flipping = chop
        if recent_signals and len(recent_signals) >= 4:
            flips = 0
            for i in range(1, len(recent_signals)):
                if recent_signals[i] != recent_signals[i-1]:
                    flips += 1
            if flips >= 3:
                chop_score += 20

        return {
            'should_filter': chop_score >= 50,
            'chop_score': chop_score,
            'reason': 'High chop — signals unreliable' if chop_score >= 50 else 'Market clear for signals',
        }


class ModelVersionTracker:
    """Track AI model versions and performance evolution."""

    CURRENT_VERSION = 'V8.0'

    def __init__(self, data_dir='data'):
        self._log_file = os.path.join(data_dir, 'model_versions.json')
        self._versions = self._load()

    def _load(self):
        try:
            if os.path.exists(self._log_file):
                with open(self._log_file, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._log_file), exist_ok=True)
            with open(self._log_file, 'w') as f:
                json.dump(self._versions[-50:], f, indent=2)
        except:
            pass

    def log_version(self, version, changes, winrate=None):
        self._versions.append({
            'version': version,
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'changes': changes,
            'winrate_at_release': winrate,
        })
        self._save()

    def get_history(self):
        return self._versions[-10:]
