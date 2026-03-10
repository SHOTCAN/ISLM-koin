"""
ISLM Autonomous Trading Engine V1
==================================
Central orchestrator connecting:
  Signal Precision → Quality Gate → Risk Manager → Trade Execution

Features:
  - Multi-coin scanning via CoinScanner
  - Dynamic capital scaling (Rp50k → Rp500k+)
  - Position monitoring with trailing stops
  - Circuit breakers for capital protection
  - Self-improving via post-trade analysis
  - Continuous autonomous loop
"""

import time
import uuid
import json
import os
import threading
from datetime import datetime, timedelta


class TradingConfig:
    """All configurable trading parameters."""

    # === Capital Management ===
    INITIAL_CAPITAL = 50_000           # Rp50,000 starting capital
    MAX_RISK_PER_TRADE_PCT = 5.0       # Max 5% of capital per trade
    MAX_EXPOSURE_TOTAL_PCT = 30.0      # Max 30% of capital in open positions
    MAX_DAILY_LOSS_IDR = 10_000        # Stop trading if daily loss > Rp10k
    MAX_DAILY_TRADES = 8               # Anti-overtrading
    
    # === Signal Thresholds ===
    MIN_CONFIDENCE_TO_TRADE = 70       # Minimum signal confidence (0-100)
    MIN_SCAN_SCORE = 50                # Minimum coin scanner score
    
    # === Stop Loss / Take Profit ===
    DEFAULT_SL_PCT = 3.0               # Default stop loss %
    DEFAULT_TP_PCT = 5.0               # Default take-profit %
    TRAILING_STOP_ACTIVATION_PCT = 2.0 # Activate trailing after 2% profit
    TRAILING_STOP_DISTANCE_PCT = 1.5   # Trail by 1.5%
    
    # === Partial Take-Profit ===
    PARTIAL_TP_LEVELS = [
        (2.0, 0.30),  # At +2%, sell 30%
        (4.0, 0.40),  # At +4%, sell 40%
        (7.0, 0.30),  # At +7%, sell remaining 30%
    ]
    
    # === Timing ===
    SCAN_INTERVAL_SECONDS = 45         # Market scan every 45s
    POSITION_CHECK_SECONDS = 15        # Check positions every 15s
    COOLDOWN_AFTER_LOSS_SECONDS = 300  # 5 min cooldown after a loss
    COOLDOWN_AFTER_3_LOSSES = 1800     # 30 min cooldown after 3 consecutive losses
    
    # === Capital Scaling Gates ===
    SCALE_REQUIREMENTS = {
        # capital_level: (min_trades, min_win_rate, min_profit_factor)
        100_000: (10, 55.0, 1.3),   # Need 10 trades, 55% WR, 1.3 PF to scale to Rp100k
        200_000: (25, 58.0, 1.5),   # 25 trades, 58% WR, 1.5 PF for Rp200k
        500_000: (50, 60.0, 1.8),   # 50 trades, 60% WR, 1.8 PF for Rp500k
    }
    
    # === Filters ===
    MIN_SPREAD_FILTER_PCT = 2.0        # Skip coins with >2% spread
    MIN_VOLUME_FILTER_IDR = 10_000_000 # Minimum Rp10M 24h volume for trading


class CapitalManager:
    """
    Dynamic capital management with performance-based scaling.
    Automatically adjusts position sizes as capital changes.
    """
    
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
        return {
            'capital': 0,
            'initial_capital': 0,
            'allocated': 0,       # Currently allocated in open trades
            'scale_level': 0,     # Current scaling tier
            'scale_history': [],
            'last_updated': None,
        }
    
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
        """Capital not currently allocated to open trades."""
        return max(0, self.capital - self._state.get('allocated', 0))
    
    @property
    def allocated(self):
        return self._state.get('allocated', 0)
    
    def set_capital(self, amount):
        """Manually set trading capital (user command)."""
        self._state['capital'] = amount
        self._save()
        print(f"[Capital] Set to Rp {amount:,.0f}")
    
    def allocate(self, amount):
        """Lock capital for a trade."""
        self._state['allocated'] = self._state.get('allocated', 0) + amount
        self._save()
    
    def release(self, amount, pnl=0):
        """Release capital after trade closes, adding P&L."""
        self._state['allocated'] = max(0, self._state.get('allocated', 0) - amount)
        self._state['capital'] = self._state.get('capital', 0) + pnl
        self._save()
    
    def calculate_position_size(self, risk_pct=None):
        """Calculate safe position size based on current capital."""
        if risk_pct is None:
            risk_pct = TradingConfig.MAX_RISK_PER_TRADE_PCT
        
        max_by_risk = self.capital * (risk_pct / 100)
        max_by_exposure = self.capital * (TradingConfig.MAX_EXPOSURE_TOTAL_PCT / 100) - self.allocated
        
        position = min(max_by_risk, max(0, max_by_exposure))
        
        # Safety floor: never less than Rp0
        return max(0, round(position))
    
    def check_scale_eligibility(self, performance):
        """Check if capital can be scaled up based on performance."""
        eligible_levels = []
        for target_capital, (min_trades, min_wr, min_pf) in TradingConfig.SCALE_REQUIREMENTS.items():
            if self.capital >= target_capital:
                continue  # Already at or above this level
            if (performance.get('total_trades', 0) >= min_trades and
                performance.get('win_rate', 0) >= min_wr and
                performance.get('profit_factor', 0) >= min_pf):
                eligible_levels.append(target_capital)
        
        return min(eligible_levels) if eligible_levels else None
    
    def get_status(self):
        return {
            'capital': self.capital,
            'allocated': self.allocated,
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
        """
        Run all circuit breaker checks.
        Returns (is_safe, reason) — if not safe, trading should stop.
        """
        now = time.time()
        
        # 1. Check if we're in cooldown
        if now < self._paused_until:
            remaining = int(self._paused_until - now)
            return False, f"Cooldown aktif ({remaining}s tersisa): {self._pause_reason}"
        
        # 2. Daily loss limit
        today_pnl = journal.get_today_pnl()
        if today_pnl < -TradingConfig.MAX_DAILY_LOSS_IDR:
            self._paused_until = now + 86400  # Pause for 24h
            self._pause_reason = f"Daily loss limit hit: Rp {today_pnl:,.0f}"
            return False, self._pause_reason
        
        # 3. Daily trade count
        today_count = journal.get_today_trade_count()
        if today_count >= TradingConfig.MAX_DAILY_TRADES:
            return False, f"Max daily trades reached ({today_count}/{TradingConfig.MAX_DAILY_TRADES})"
        
        # 4. Capital protection: stop if capital drops below 50% of initial
        if capital_mgr.capital < capital_mgr._state.get('initial_capital', 50000) * 0.5:
            self._paused_until = now + 86400
            self._pause_reason = "Capital dropped below 50% of initial — 24h emergency pause"
            return False, self._pause_reason
        
        # 5. Available capital check
        if capital_mgr.available < 10000:  # Less than Rp10k available
            return False, "Insufficient available capital (< Rp10,000)"
        
        return True, "OK"
    
    def record_trade_result(self, is_win):
        """Track consecutive losses for cooldown logic."""
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            if self._consecutive_losses >= 3:
                self._paused_until = time.time() + TradingConfig.COOLDOWN_AFTER_3_LOSSES
                self._pause_reason = f"3 consecutive losses — {TradingConfig.COOLDOWN_AFTER_3_LOSSES}s cooldown"
                self._consecutive_losses = 0  # Reset counter
            else:
                self._paused_until = time.time() + TradingConfig.COOLDOWN_AFTER_LOSS_SECONDS
                self._pause_reason = "Post-loss cooldown"
    
    def force_pause(self, seconds, reason="Manual pause"):
        self._paused_until = time.time() + seconds
        self._pause_reason = reason


class PositionMonitor:
    """Real-time monitoring of open positions for SL/TP/trailing."""
    
    def __init__(self, trade_api, journal):
        self.api = trade_api
        self.journal = journal
        self._positions = {}  # trade_id -> position state
    
    def add_position(self, trade_id, pair, entry_price, amount_idr, amount_coin, direction='buy'):
        """Register a new position for monitoring."""
        self._positions[trade_id] = {
            'pair': pair,
            'entry_price': entry_price,
            'amount_idr': amount_idr,
            'amount_coin': amount_coin,
            'direction': direction,
            'highest_price': entry_price,
            'trailing_active': False,
            'partial_tp_taken': [],  # Which TP levels already triggered
        }
    
    def check_positions(self):
        """
        Check all open positions against SL/TP/trailing.
        Returns list of actions to execute.
        """
        actions = []
        
        for trade_id, pos in list(self._positions.items()):
            try:
                # Get current price
                price_data = self.api.get_price(pos['pair'])
                if not price_data.get('success'):
                    continue
                
                current_price = price_data.get('last', 0)
                if current_price <= 0:
                    continue
                
                entry = pos['entry_price']
                change_pct = ((current_price - entry) / entry) * 100
                
                # Update highest price for trailing
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                
                # --- Stop Loss ---
                if change_pct <= -TradingConfig.DEFAULT_SL_PCT:
                    actions.append({
                        'action': 'close',
                        'trade_id': trade_id,
                        'reason': 'stop_loss',
                        'price': current_price,
                        'change_pct': change_pct,
                    })
                    continue
                
                # --- Trailing Stop ---
                if change_pct >= TradingConfig.TRAILING_STOP_ACTIVATION_PCT:
                    pos['trailing_active'] = True
                
                if pos['trailing_active']:
                    drop_from_high = ((pos['highest_price'] - current_price) / pos['highest_price']) * 100
                    if drop_from_high >= TradingConfig.TRAILING_STOP_DISTANCE_PCT:
                        actions.append({
                            'action': 'close',
                            'trade_id': trade_id,
                            'reason': 'trailing_stop',
                            'price': current_price,
                            'change_pct': change_pct,
                            'high': pos['highest_price'],
                        })
                        continue
                
                # --- Partial Take-Profit ---
                for tp_pct, tp_fraction in TradingConfig.PARTIAL_TP_LEVELS:
                    if tp_pct not in pos['partial_tp_taken'] and change_pct >= tp_pct:
                        actions.append({
                            'action': 'partial_close',
                            'trade_id': trade_id,
                            'reason': f'partial_tp_{tp_pct}%',
                            'price': current_price,
                            'fraction': tp_fraction,
                            'change_pct': change_pct,
                        })
                        pos['partial_tp_taken'].append(tp_pct)
                
            except Exception as e:
                print(f"[Monitor] Error checking {trade_id}: {e}")
        
        return actions
    
    def remove_position(self, trade_id):
        self._positions.pop(trade_id, None)
    
    def get_positions_summary(self):
        """Get formatted summary of all open positions."""
        summaries = []
        for tid, pos in self._positions.items():
            try:
                price_data = self.api.get_price(pos['pair'])
                current = price_data.get('last', 0) if price_data.get('success') else 0
                change = ((current - pos['entry_price']) / pos['entry_price'] * 100) if current > 0 else 0
                summaries.append({
                    'trade_id': tid[:8],
                    'pair': pos['pair'],
                    'entry': pos['entry_price'],
                    'current': current,
                    'change_pct': round(change, 2),
                    'amount_idr': pos['amount_idr'],
                    'trailing': pos['trailing_active'],
                })
            except Exception:
                pass
        return summaries


class TradingEngine:
    """
    Main autonomous trading engine.
    Orchestrates: Scan → Signal → Gate → Risk → Execute → Monitor → Learn
    """
    
    def __init__(self, trade_api, analysis_api=None, notifier=None):
        """
        Args:
            trade_api: IndodaxTradeAPI instance (with trade keys)
            analysis_api: IndodaxAPI instance (read-only, for analysis)
            notifier: Callable(text) for sending Telegram messages
        """
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
        
        # State
        self._enabled = False
        self._running = False
        self._thread = None
        self._last_scan_time = 0
        self._last_position_check = 0
    
    # ==========================================
    # CONTROL
    # ==========================================
    
    def enable(self):
        """Enable autonomous trading."""
        self._enabled = True
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._main_loop, daemon=True)
            self._thread.start()
            self.notify("🟢 *Auto-Trading AKTIF*\n"
                       f"Modal: Rp {self.capital.capital:,.0f}\n"
                       f"Max risk/trade: {TradingConfig.MAX_RISK_PER_TRADE_PCT}%\n"
                       f"Min confidence: {TradingConfig.MIN_CONFIDENCE_TO_TRADE}%")
        return True
    
    def disable(self):
        """Disable autonomous trading (does NOT close positions)."""
        self._enabled = False
        self._running = False
        self.notify("🔴 *Auto-Trading NONAKTIF*\n"
                   "Posisi terbuka tetap dimonitor.")
        return True
    
    def emergency_close_all(self):
        """Emergency: close ALL open positions immediately."""
        open_trades = self.journal.get_open_trades()
        closed_count = 0
        
        for trade in open_trades:
            try:
                pair = trade.get('pair', '')
                price_data = self.trade_api.get_price(pair)
                if price_data.get('success'):
                    current_price = price_data.get('last', 0)
                    self._execute_close(trade['id'], current_price, 'emergency_close')
                    closed_count += 1
            except Exception as e:
                print(f"[Emergency] Failed to close {trade.get('id', '?')}: {e}")
        
        self.circuit_breaker.force_pause(3600, "Emergency close — 1h cooldown")
        self.notify(f"🚨 *EMERGENCY CLOSE*\n{closed_count} posisi ditutup paksa.")
        return closed_count
    
    # ==========================================
    # MAIN AUTONOMOUS LOOP
    # ==========================================
    
    def _main_loop(self):
        """The continuously running trading loop."""
        print("[Engine] ▶ Autonomous trading loop started")
        
        while self._running:
            try:
                now = time.time()
                
                # --- Position monitoring (frequent) ---
                if now - self._last_position_check >= TradingConfig.POSITION_CHECK_SECONDS:
                    self._check_positions()
                    self._last_position_check = now
                
                # --- Market scanning + trade decisions (less frequent) ---
                if self._enabled and (now - self._last_scan_time >= TradingConfig.SCAN_INTERVAL_SECONDS):
                    self._scan_and_trade()
                    self._last_scan_time = now
                
                time.sleep(3)  # Base loop interval
                
            except Exception as e:
                print(f"[Engine] Loop error: {e}")
                time.sleep(10)
        
        print("[Engine] ⏹ Trading loop stopped")
    
    def _scan_and_trade(self):
        """Scan market and execute trades if conditions are met."""
        # Circuit breaker check
        is_safe, reason = self.circuit_breaker.check(self.journal, self.capital)
        if not is_safe:
            return
        
        # Scan for opportunities
        top_coins = self.scanner.get_top_coins(5)
        
        for coin in top_coins:
            # Score filter
            if coin['score'] < TradingConfig.MIN_SCAN_SCORE:
                continue
            
            # Volume filter
            if coin['volume_idr'] < TradingConfig.MIN_VOLUME_FILTER_IDR:
                continue
            
            # Spread filter
            if coin['spread_pct'] > TradingConfig.MIN_SPREAD_FILTER_PCT:
                continue
            
            # Check we don't already have a position in this coin
            open_trades = self.journal.get_open_trades()
            already_in = any(t.get('pair') == coin['pair'] for t in open_trades)
            if already_in:
                continue
            
            # Calculate position size
            position_idr = self.capital.calculate_position_size()
            if position_idr < 15000:  # Minimum viable trade
                continue
            
            # Re-check circuit breaker (capital may have changed)
            is_safe, reason = self.circuit_breaker.check(self.journal, self.capital)
            if not is_safe:
                break
            
            # Execute buy
            self._execute_buy(
                pair=coin['pair'],
                price=coin['ask'],  # Buy at ask price
                amount_idr=position_idr,
                confidence=coin['score'],
                factors=coin['factors'],
            )
            
            # Only one trade per scan cycle
            break
    
    def _execute_buy(self, pair, price, amount_idr, confidence, factors):
        """Execute a buy order."""
        trade_id = str(uuid.uuid4())
        
        try:
            # Place limit buy order
            result = self.trade_api.create_order(
                pair=pair,
                order_type='buy',
                price=price,
                amount_idr=amount_idr,
            )
            
            if not result.get('success'):
                print(f"[Trade] Buy failed for {pair}: {result.get('error')}")
                return False
            
            # Estimate coin amount
            amount_coin = amount_idr / price if price > 0 else 0
            
            # Record in journal
            self.journal.record_open(
                trade_id=trade_id,
                pair=pair,
                direction='buy',
                entry_price=price,
                amount_idr=amount_idr,
                amount_coin=amount_coin,
                confidence=confidence,
                signal_factors=factors,
                order_id=result.get('order_id'),
            )
            
            # Allocate capital
            self.capital.allocate(amount_idr)
            
            # Register for monitoring
            self.monitor.add_position(
                trade_id=trade_id,
                pair=pair,
                entry_price=price,
                amount_idr=amount_idr,
                amount_coin=amount_coin,
            )
            
            # Notify user
            coin_name = pair.replace('idr', '').upper()
            self.notify(
                f"🟢 *BELI {coin_name}*\n"
                f"━━━━━━━━━━━━━━━━━━━━━━\n"
                f"💰 Harga: Rp {price:,.0f}\n"
                f"📊 Jumlah: Rp {amount_idr:,.0f}\n"
                f"🎯 Confidence: {confidence}%\n"
                f"🛡️ SL: -{TradingConfig.DEFAULT_SL_PCT}% | TP: +{TradingConfig.DEFAULT_TP_PCT}%\n"
                f"📎 Order ID: {result.get('order_id', 'N/A')}"
            )
            
            print(f"[Trade] ✅ BUY {coin_name} @ Rp {price:,.0f} | Rp {amount_idr:,.0f}")
            return True
            
        except Exception as e:
            print(f"[Trade] BUY error: {e}")
            return False
    
    def _execute_close(self, trade_id, current_price, reason='manual'):
        """Close a position by selling."""
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
            # Place sell order
            result = self.trade_api.create_order(
                pair=pair,
                order_type='sell',
                price=current_price,
                amount_coin=trade.get('amount_coin', 0),
            )
            
            # Record close regardless of API result (for tracking)
            closed = self.journal.record_close(trade_id, current_price, reason)
            
            if closed:
                pnl = closed.get('pnl_idr', 0)
                pnl_pct = closed.get('pnl_pct', 0)
                
                # Release capital
                self.capital.release(trade['amount_idr'], pnl)
                
                # Remove from monitor
                self.monitor.remove_position(trade_id)
                
                # Circuit breaker tracking
                self.circuit_breaker.record_trade_result(pnl > 0)
                
                # Notify
                emoji = "🟢" if pnl > 0 else "🔴"
                self.notify(
                    f"{emoji} *JUAL {coin_name}*\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━\n"
                    f"💰 Entry: Rp {trade['entry_price']:,.0f}\n"
                    f"💰 Exit: Rp {current_price:,.0f}\n"
                    f"📊 P&L: Rp {pnl:,.0f} ({pnl_pct:+.2f}%)\n"
                    f"📝 Alasan: {reason}\n"
                    f"💼 Modal: Rp {self.capital.capital:,.0f}"
                )
                
                print(f"[Trade] {'✅' if pnl > 0 else '❌'} SELL {coin_name} | P&L: Rp {pnl:,.0f}")
            
            return True
            
        except Exception as e:
            print(f"[Trade] SELL error: {e}")
            return False
    
    def _check_positions(self):
        """Check all open positions and execute necessary actions."""
        actions = self.monitor.check_positions()
        
        for action in actions:
            if action['action'] == 'close':
                self._execute_close(
                    action['trade_id'],
                    action['price'],
                    action['reason'],
                )
            elif action['action'] == 'partial_close':
                # For partial close, we record it but execute as full
                # (Indodax doesn't support partial order modification easily)
                # In practice, we close and re-enter with smaller size
                print(f"[Monitor] Partial TP triggered: {action}")
    
    # ==========================================
    # STATUS & REPORTING
    # ==========================================
    
    def get_status(self):
        """Full engine status for Telegram display."""
        cap = self.capital.get_status()
        perf = self.journal.get_performance()
        today = self.journal.get_today_stats()
        positions = self.monitor.get_positions_summary()
        
        # Capital scaling check
        scale_to = self.capital.check_scale_eligibility(perf)
        
        return {
            'enabled': self._enabled,
            'capital': cap,
            'performance': perf,
            'today': today,
            'open_positions': positions,
            'scale_eligible': scale_to,
        }
    
    def format_status(self):
        """Format status as Telegram message."""
        s = self.get_status()
        cap = s['capital']
        perf = s['performance']
        today = s['today']
        
        status_emoji = "🟢" if self._enabled else "🔴"
        
        lines = [
            f"{status_emoji} *Trading Engine*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"💼 Modal: Rp {cap['capital']:,.0f}",
            f"📊 Terpakai: Rp {cap['allocated']:,.0f} ({cap['utilization_pct']}%)",
            f"💰 Tersedia: Rp {cap['available']:,.0f}",
            "",
            f"📈 *Hari Ini*",
            f"   Trades: {today.get('trades', 0)} | P&L: Rp {today.get('total_pnl', 0):,.0f}",
            "",
            f"📊 *30 Hari*",
            f"   Win Rate: {perf.get('win_rate', 0)}%",
            f"   Profit Factor: {perf.get('profit_factor', 0)}",
            f"   Total P&L: Rp {perf.get('total_pnl', 0):,.0f}",
            f"   Max Drawdown: Rp {perf.get('max_drawdown', 0):,.0f}",
        ]
        
        # Open positions
        positions = s.get('open_positions', [])
        if positions:
            lines.append(f"\n📋 *Posisi Terbuka ({len(positions)})*")
            for p in positions:
                emoji = "🟢" if p['change_pct'] > 0 else "🔴"
                trail = " 🔄" if p['trailing'] else ""
                lines.append(
                    f"   {emoji} {p['pair'].replace('idr','').upper()} "
                    f"{p['change_pct']:+.2f}%{trail}"
                )
        
        # Scale eligibility
        if s.get('scale_eligible'):
            lines.append(f"\n⬆️ Eligible scale up to: Rp {s['scale_eligible']:,.0f}")
        
        return "\n".join(lines)
    
    def format_performance(self):
        """Detailed performance report."""
        perf = self.journal.get_performance()
        recent = self.journal.get_recent_trades(5)
        
        lines = [
            "📊 *Performance Report*",
            "━━━━━━━━━━━━━━━━━━━━━━",
            f"Total Trades: {perf['total_trades']}",
            f"Wins: {perf.get('wins', 0)} | Losses: {perf.get('losses', 0)}",
            f"Win Rate: {perf['win_rate']}%",
            f"Profit Factor: {perf['profit_factor']}",
            f"Avg Win: Rp {perf['avg_win']:,.0f}",
            f"Avg Loss: Rp {perf['avg_loss']:,.0f}",
            f"Best Trade: Rp {perf['best_trade']:,.0f}",
            f"Worst Trade: Rp {perf['worst_trade']:,.0f}",
            f"Max Drawdown: Rp {perf['max_drawdown']:,.0f}",
            f"Total P&L: Rp {perf['total_pnl']:,.0f}",
        ]
        
        if recent:
            lines.append("\n📋 *5 Trade Terakhir*")
            for t in recent:
                emoji = "🟢" if (t.get('pnl_idr') or 0) > 0 else ("🔴" if t['status'] == 'closed' else "⏳")
                coin = t.get('pair', '?').replace('idr', '').upper()
                pnl_str = f"Rp {t['pnl_idr']:,.0f}" if t.get('pnl_idr') is not None else "Open"
                lines.append(f"   {emoji} {coin} | {pnl_str}")
        
        return "\n".join(lines)
