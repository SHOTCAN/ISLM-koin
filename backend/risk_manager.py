"""
ISLM Monitor — Institutional Risk Manager V8
=============================================
Professional risk management system:
  - Multi-entry scaling plan (40/30/30)
  - ATR-based dynamic stop loss
  - Trailing stop with activation threshold
  - Take profit partial (3 levels: 30/40/30)
  - Risk-Reward ratio enforcement (min 1.5)
  - Kelly Criterion position sizing
  - Exposure cap per asset & total portfolio
  - Smart DCA (only if structure valid)
  - Full trade plan generator
"""

import statistics


class RiskManager:
    """Institutional-grade risk management for ISLM trading."""

    # Default risk parameters
    MAX_RISK_PER_TRADE = 0.02   # 2% max risk per trade
    MIN_RR_RATIO = 1.5          # Minimum risk-reward
    MAX_EXPOSURE_SINGLE = 0.25  # 25% max in single asset
    MAX_EXPOSURE_TOTAL = 0.50   # 50% max total exposure
    SCALING_PLAN = [0.40, 0.30, 0.30]  # Entry scaling: 40/30/30
    TP_PLAN = [0.30, 0.40, 0.30]       # TP scaling: 30/40/30

    @staticmethod
    def calculate_entry_zones(price, support, resistance, atr=None):
        """Calculate 3 optimal entry zones based on support + ATR."""
        if not support or not resistance or price <= 0:
            return {'valid': False}

        # Zone 1: Near current price (most aggressive)
        zone1 = price * 0.998  # 0.2% below current

        # Zone 2: Near support (moderate)
        zone2 = support * 1.005  # Just above support

        # Zone 3: Deep support (conservative, uses ATR if available)
        if atr and atr > 0:
            zone3 = support - atr * 0.5  # Half ATR below support
        else:
            zone3 = support * 0.99  # 1% below support

        return {
            'valid': True,
            'zones': [
                {'price': round(zone1, 2), 'allocation': '40%', 'label': 'Zone 1 — Agresif'},
                {'price': round(zone2, 2), 'allocation': '30%', 'label': 'Zone 2 — Support'},
                {'price': round(zone3, 2), 'allocation': '30%', 'label': 'Zone 3 — Deep Buy'},
            ],
            'avg_entry': round(zone1 * 0.4 + zone2 * 0.3 + zone3 * 0.3, 2),
            'support_ref': support,
            'resistance_ref': resistance,
        }

    @staticmethod
    def calculate_dynamic_sl(entry, atr, structure_low=None, multiplier=1.5):
        """ATR-based dynamic stop loss with structure awareness."""
        if entry <= 0 or not atr or atr <= 0:
            return {'valid': False}

        atr_sl = entry - atr * multiplier
        structure_sl = structure_low * 0.995 if structure_low else atr_sl

        # Use the tighter of ATR-SL and structure-SL
        sl = max(atr_sl, structure_sl)

        risk_pct = (entry - sl) / entry * 100

        return {
            'valid': True,
            'stop_loss': round(sl, 2),
            'atr_based': round(atr_sl, 2),
            'structure_based': round(structure_sl, 2) if structure_low else None,
            'risk_pct': round(risk_pct, 2),
            'atr_multiplier': multiplier,
        }

    @staticmethod
    def calculate_trailing_stop(entry, current, atr, activation_pct=3.0, trail_atr_mult=2.0):
        """Trailing stop: activates after X% profit, trails by ATR."""
        if entry <= 0 or current <= 0 or not atr:
            return {'active': False}

        profit_pct = (current - entry) / entry * 100

        if profit_pct < activation_pct:
            return {
                'active': False,
                'profit_pct': round(profit_pct, 2),
                'activation_at': round(entry * (1 + activation_pct / 100), 2),
                'status': f'Aktif setelah +{activation_pct}% (masih {profit_pct:.1f}%)',
            }

        trail_sl = current - atr * trail_atr_mult

        return {
            'active': True,
            'trailing_sl': round(trail_sl, 2),
            'profit_pct': round(profit_pct, 2),
            'locked_profit': round((trail_sl - entry) / entry * 100, 2),
            'status': f'AKTIF — SL trailing di Rp {trail_sl:,.0f}',
        }

    @staticmethod
    def calculate_tp_levels(entry, resistance, atr=None):
        """3-level take profit plan: 30/40/30 split."""
        if entry <= 0:
            return {'valid': False}

        # TP1: First resistance
        tp1 = resistance if resistance else entry * 1.03

        # TP2: 1.5x the TP1 distance
        dist = tp1 - entry
        tp2 = entry + dist * 1.5

        # TP3: 2.5x distance (moonshot)
        tp3 = entry + dist * 2.5

        return {
            'valid': True,
            'levels': [
                {'price': round(tp1, 2), 'sell_pct': '30%', 'label': 'TP1 — Resistance',
                 'rr': round((tp1 - entry) / max(entry * 0.02, 1), 1)},
                {'price': round(tp2, 2), 'sell_pct': '40%', 'label': 'TP2 — Extension',
                 'rr': round((tp2 - entry) / max(entry * 0.02, 1), 1)},
                {'price': round(tp3, 2), 'sell_pct': '30%', 'label': 'TP3 — Moonshot',
                 'rr': round((tp3 - entry) / max(entry * 0.02, 1), 1)},
            ],
        }

    @staticmethod
    def optimal_rr_ratio(entry, stop_loss, take_profit):
        """Calculate risk-reward ratio. Must be >= 1.5."""
        if entry <= 0 or stop_loss <= 0 or take_profit <= 0:
            return {'valid': False}

        risk = entry - stop_loss
        reward = take_profit - entry

        if risk <= 0:
            return {'valid': False, 'reason': 'SL above entry'}

        rr = reward / risk

        return {
            'valid': True,
            'rr_ratio': round(rr, 2),
            'risk': round(risk, 2),
            'reward': round(reward, 2),
            'acceptable': rr >= RiskManager.MIN_RR_RATIO,
            'grade': 'EXCELLENT' if rr >= 3 else ('GOOD' if rr >= 2 else ('OK' if rr >= 1.5 else 'POOR')),
        }

    @staticmethod
    def kelly_position_size(winrate, avg_win, avg_loss, capital, max_kelly=0.25):
        """Kelly Criterion for optimal position sizing."""
        if winrate <= 0 or avg_win <= 0 or avg_loss >= 0:
            return {'valid': False}

        win_prob = winrate / 100
        loss_prob = 1 - win_prob
        b = avg_win / abs(avg_loss)  # Win/loss ratio

        kelly = win_prob - (loss_prob / b) if b > 0 else 0
        kelly = max(0, min(kelly, max_kelly))  # Cap Kelly

        # Half-Kelly for safety
        half_kelly = kelly / 2

        position_size = capital * half_kelly

        return {
            'valid': True,
            'full_kelly': round(kelly * 100, 1),
            'half_kelly': round(half_kelly * 100, 1),
            'position_size': round(position_size, 0),
            'capital': capital,
            'win_loss_ratio': round(b, 2),
        }

    @staticmethod
    def check_exposure(current_islm_value, total_portfolio, limits=None):
        """Check if position size exceeds exposure limits."""
        max_single = (limits or {}).get('max_single', RiskManager.MAX_EXPOSURE_SINGLE)
        max_total = (limits or {}).get('max_total', RiskManager.MAX_EXPOSURE_TOTAL)

        if total_portfolio <= 0:
            return {'valid': False}

        exposure_pct = current_islm_value / total_portfolio

        return {
            'valid': True,
            'exposure_pct': round(exposure_pct * 100, 1),
            'max_allowed': round(max_single * 100, 1),
            'over_exposed': exposure_pct > max_single,
            'action': 'REDUCE' if exposure_pct > max_single else 'OK',
        }

    @staticmethod
    def smart_dca_check(current_price, avg_entry, structure_trend, rsi, support):
        """Only DCA if market structure supports it."""
        if not avg_entry or avg_entry <= 0 or current_price <= 0:
            return {'should_dca': False, 'reason': 'No position'}

        # Price must be below average entry
        below_avg = current_price < avg_entry
        discount_pct = (avg_entry - current_price) / avg_entry * 100

        # Structure must be valid (UPTREND or SIDEWAYS, not DOWNTREND)
        valid_structure = structure_trend in ('UPTREND', 'SIDEWAYS', 'CONTRACTING')

        # RSI must be oversold (<40)
        rsi_ok = rsi < 40

        # Price near support
        near_support = current_price <= support * 1.02 if support else False

        should_dca = below_avg and valid_structure and (rsi_ok or near_support)

        return {
            'should_dca': should_dca,
            'below_entry': below_avg,
            'discount_pct': round(discount_pct, 1),
            'valid_structure': valid_structure,
            'structure': structure_trend,
            'rsi_ok': rsi_ok,
            'rsi': rsi,
            'near_support': near_support,
            'reason': 'Structure valid + discount' if should_dca else
                      'Bad structure' if not valid_structure else
                      'RSI too high' if not rsi_ok else 'Price above avg entry',
        }

    @staticmethod
    def generate_trade_plan(price, support, resistance, atr, structure_trend,
                           rsi, signal_label, confidence, winrate_data=None,
                           capital=None):
        """Generate complete institutional trade plan."""

        # Entry zones
        entries = RiskManager.calculate_entry_zones(price, support, resistance, atr)

        # Stop loss
        sl_data = RiskManager.calculate_dynamic_sl(
            entries['avg_entry'] if entries.get('valid') else price,
            atr,
            structure_low=support
        )

        # Take profit
        tp_data = RiskManager.calculate_tp_levels(
            entries['avg_entry'] if entries.get('valid') else price,
            resistance, atr
        )

        # Risk-reward
        rr = {'valid': False}
        if sl_data.get('valid') and tp_data.get('valid') and tp_data['levels']:
            avg_tp = tp_data['levels'][1]['price']  # Use TP2 for RR calc
            rr = RiskManager.optimal_rr_ratio(
                entries['avg_entry'] if entries.get('valid') else price,
                sl_data['stop_loss'], avg_tp
            )

        # Position sizing (if capital and winrate provided)
        sizing = {'valid': False}
        if capital and winrate_data and winrate_data.get('valid'):
            sizing = RiskManager.kelly_position_size(
                winrate_data['winrate'],
                winrate_data['avg_win'],
                winrate_data['avg_loss'],
                capital
            )

        return {
            'signal': signal_label,
            'confidence': confidence,
            'price': price,
            'entries': entries,
            'stop_loss': sl_data,
            'take_profit': tp_data,
            'risk_reward': rr,
            'position_sizing': sizing,
            'structure': structure_trend,
            'rsi': rsi,
            'winrate': winrate_data if winrate_data else {'valid': False},
            'acceptable': rr.get('acceptable', False),
            'grade': rr.get('grade', 'N/A'),
        }
