"""
Trade Journal — Persistent Trade Log & Performance Metrics
==========================================================
- JSON-backed trade history
- Win rate, profit factor, drawdown tracking
- Per-coin and per-strategy breakdown
- Used by Adaptive Learner for self-improvement
"""

import json
import os
import time
from datetime import datetime, timedelta
from collections import defaultdict


class TradeJournal:
    """Persistent log of every trade with performance metrics."""

    def __init__(self, data_dir='data'):
        os.makedirs(data_dir, exist_ok=True)
        self._filepath = os.path.join(data_dir, 'trade_journal.json')
        self._data = self._load()

    def _load(self):
        try:
            if os.path.exists(self._filepath):
                with open(self._filepath, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {'trades': [], 'daily_stats': {}, 'version': '1.0'}

    def _save(self):
        try:
            with open(self._filepath, 'w') as f:
                json.dump(self._data, f, indent=2, default=str)
        except Exception as e:
            print(f"[Journal] Save failed: {e}")

    def record_open(self, trade_id, pair, direction, entry_price, amount_idr,
                    amount_coin, confidence, signal_factors, order_id=None):
        """Record a new trade opening."""
        trade = {
            'id': trade_id,
            'pair': pair,
            'direction': direction,  # 'buy' or 'sell'
            'entry_price': entry_price,
            'amount_idr': amount_idr,
            'amount_coin': amount_coin,
            'confidence': confidence,
            'signal_factors': signal_factors,
            'order_id': order_id,
            'status': 'open',
            'opened_at': datetime.utcnow().isoformat(),
            'closed_at': None,
            'exit_price': None,
            'pnl_idr': None,
            'pnl_pct': None,
            'exit_reason': None,
            'duration_minutes': None,
        }
        self._data['trades'].append(trade)
        self._save()
        return trade

    def record_close(self, trade_id, exit_price, exit_reason='manual'):
        """Record a trade closure with P&L."""
        for trade in self._data['trades']:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['status'] = 'closed'
                trade['exit_price'] = exit_price
                trade['exit_reason'] = exit_reason
                trade['closed_at'] = datetime.utcnow().isoformat()

                # Calculate P&L
                if trade['direction'] == 'buy':
                    pnl_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
                    pnl_idr = trade['amount_idr'] * (pnl_pct / 100)
                else:
                    pnl_pct = ((trade['entry_price'] - exit_price) / trade['entry_price']) * 100
                    pnl_idr = trade['amount_idr'] * (pnl_pct / 100)

                trade['pnl_pct'] = round(pnl_pct, 4)
                trade['pnl_idr'] = round(pnl_idr, 2)

                # Duration
                try:
                    opened = datetime.fromisoformat(trade['opened_at'])
                    closed = datetime.fromisoformat(trade['closed_at'])
                    trade['duration_minutes'] = round((closed - opened).total_seconds() / 60, 1)
                except Exception:
                    pass

                # Update daily stats
                today = datetime.utcnow().strftime('%Y-%m-%d')
                if today not in self._data['daily_stats']:
                    self._data['daily_stats'][today] = {
                        'trades': 0, 'wins': 0, 'losses': 0,
                        'total_pnl': 0, 'max_drawdown': 0
                    }
                ds = self._data['daily_stats'][today]
                ds['trades'] += 1
                ds['total_pnl'] = round(ds['total_pnl'] + pnl_idr, 2)
                if pnl_idr > 0:
                    ds['wins'] += 1
                else:
                    ds['losses'] += 1
                    ds['max_drawdown'] = min(ds['max_drawdown'], ds['total_pnl'])

                self._save()
                return trade
        return None

    def get_open_trades(self):
        """Get all currently open trades."""
        return [t for t in self._data['trades'] if t['status'] == 'open']

    def get_closed_trades(self, days=30):
        """Get closed trades within the last N days."""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        return [
            t for t in self._data['trades']
            if t['status'] == 'closed' and (t.get('closed_at', '') >= cutoff)
        ]

    def get_today_stats(self):
        """Get today's trading statistics."""
        today = datetime.utcnow().strftime('%Y-%m-%d')
        stats = self._data['daily_stats'].get(today, {
            'trades': 0, 'wins': 0, 'losses': 0,
            'total_pnl': 0, 'max_drawdown': 0
        })
        return stats

    def get_today_trade_count(self):
        """Shortcut: how many trades executed today."""
        return self.get_today_stats().get('trades', 0)

    def get_today_pnl(self):
        """Shortcut: today's total P&L in IDR."""
        return self.get_today_stats().get('total_pnl', 0)

    def get_performance(self, days=30):
        """
        Full performance report.
        Returns win rate, profit factor, avg win, avg loss, max drawdown.
        """
        closed = self.get_closed_trades(days)
        if not closed:
            return {
                'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'avg_win': 0, 'avg_loss': 0, 'total_pnl': 0,
                'max_drawdown': 0, 'best_trade': 0, 'worst_trade': 0,
            }

        wins = [t for t in closed if (t.get('pnl_idr', 0) or 0) > 0]
        losses = [t for t in closed if (t.get('pnl_idr', 0) or 0) <= 0]

        total_win = sum(t.get('pnl_idr', 0) or 0 for t in wins)
        total_loss = abs(sum(t.get('pnl_idr', 0) or 0 for t in losses))

        pnls = [t.get('pnl_idr', 0) or 0 for t in closed]

        # Running drawdown
        peak = 0
        max_dd = 0
        cumulative = 0
        for pnl in pnls:
            cumulative += pnl
            peak = max(peak, cumulative)
            dd = peak - cumulative
            max_dd = max(max_dd, dd)

        return {
            'total_trades': len(closed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(len(wins) / len(closed) * 100, 1) if closed else 0,
            'profit_factor': round(total_win / total_loss, 2) if total_loss > 0 else 999,
            'avg_win': round(total_win / len(wins), 2) if wins else 0,
            'avg_loss': round(total_loss / len(losses), 2) if losses else 0,
            'total_pnl': round(sum(pnls), 2),
            'max_drawdown': round(max_dd, 2),
            'best_trade': round(max(pnls), 2) if pnls else 0,
            'worst_trade': round(min(pnls), 2) if pnls else 0,
        }

    def get_recent_trades(self, count=10):
        """Get last N trades (open + closed)."""
        return self._data['trades'][-count:]
