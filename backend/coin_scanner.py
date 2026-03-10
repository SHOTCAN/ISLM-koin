"""
Coin Scanner V4 — Multi-Tier Market Intelligence
==================================================
3-tier architecture for speed + precision:

  Tier 1: BROAD SCAN (every 30-45s, 1 API call)
    get_summaries() → instant pre-filter → score → top 10
    Cost: ~200ms, zero kline calls

  Tier 2: WATCHLIST (persistent across scans)
    Coins that appear in top-10 for 2+ consecutive scans = "hot"
    Tracks score velocity (rising/falling momentum)
    Coins that drop off for 3 scans = removed from watchlist

  Tier 3: READY-TO-TRADE trigger
    Watchlist coins with rising velocity + score > threshold
    → flagged as "ready for deep analysis"
    Trading engine ONLY calls get_kline on these coins

The expensive deep AI analysis (get_kline, multi-TF, manipulation
check) is NEVER called from this scanner. It's called by the
trading engine only for Tier-3 flagged coins.

This architecture means:
  - 300+ pairs scanned in <200ms
  - Only 2-3 coins get expensive deep analysis per cycle
  - Momentum shifts detected across scan intervals
"""

import time
from datetime import datetime


# Stablecoins / wrapped assets — never worth trading
BLACKLIST = {
    'usdtidr', 'usdcidr', 'busdidr', 'daiusd', 'tusdidr',
    'bidr', 'idrt', 'idrtidr', 'paxidr',
}


class CoinScanner:
    """Multi-tier market scanner with watchlist promotion."""

    # === Pre-Filter Thresholds ===
    MIN_VOLUME_IDR = 5_000_000     # Rp5M min daily volume
    MIN_PRICE_IDR = 10             # Skip dust
    MAX_SPREAD_PCT = 2.5           # Pre-filter spread
    MIN_RANGE_PCT = 0.5            # Min 24h price range %

    # === Scoring ===
    IDEAL_VOLUME_IDR = 50_000_000  # Full volume score at Rp50M
    TOP_N = 10

    # === Watchlist ===
    WATCHLIST_PROMOTE_AFTER = 2    # Promote after 2 consecutive appearances
    WATCHLIST_DEMOTE_AFTER = 3     # Remove after 3 consecutive absences
    VELOCITY_BULLISH = 3.0         # Score increase > 3 per scan = bullish velocity

    def __init__(self, api):
        self.api = api
        self._last_scan = None
        self._last_results = []
        self._scan_cooldown = 30
        self._scan_count = 0

        # Tier 2: Watchlist state
        self._watchlist = {}       # pair -> WatchlistEntry
        self._previous_prices = {} # pair -> last known price
        self._previous_scores = {} # pair -> last scan score

    def scan_market(self):
        """
        Tier 1: Broad scan + Tier 2: Watchlist update.
        Returns top candidates sorted by effective score.
        """
        now = time.time()
        if self._last_scan and (now - self._last_scan) < self._scan_cooldown:
            return self._last_results

        t0 = time.time()
        summaries = self.api.get_summaries()
        if not summaries.get('success'):
            return self._last_results

        tickers = summaries.get('tickers', {})
        candidates = []
        current_pairs = set()  # Track which pairs appear this scan
        stats = {'total': 0, 'passed': 0, 'rejected': 0}

        for pair_id, data in tickers.items():
            stats['total'] += 1
            try:
                # ========== TIER 1: INSTANT PRE-FILTER ==========
                clean_pair = pair_id.replace('_', '').lower()

                if not clean_pair.endswith('idr'):
                    continue
                if clean_pair in BLACKLIST:
                    continue

                last = float(data.get('last', 0))
                high = float(data.get('high', 0))
                low = float(data.get('low', 0))
                buy = float(data.get('buy', 0))
                sell = float(data.get('sell', 0))

                vol_idr = 0
                for vk in ['vol_idr', 'vol_base', 'volume']:
                    v = data.get(vk)
                    if v:
                        try:
                            vol_idr = float(v)
                            if vol_idr > 0:
                                break
                        except (ValueError, TypeError):
                            pass

                # Fast rejects
                if last < self.MIN_PRICE_IDR or vol_idr < self.MIN_VOLUME_IDR:
                    continue
                if buy <= 0 or sell <= 0 or sell <= buy:
                    continue

                spread_pct = ((sell - buy) / buy) * 100
                if spread_pct > self.MAX_SPREAD_PCT:
                    continue

                if high <= low or high == 0:
                    continue
                price_range = high - low
                range_pct = (price_range / low) * 100
                if range_pct < self.MIN_RANGE_PCT:
                    continue

                position_in_range = (last - low) / price_range
                if position_in_range < 0.20:  # Bottom 20% = definitely falling
                    continue

                # ========== SCORING (max ~110) ==========
                score = 0
                factors = {}

                # 1. Momentum (max 25)
                momentum = position_in_range * 25
                score += momentum
                factors['momentum'] = round(momentum, 1)

                # 2. Volume (max 20)
                vol_score = min(20, (vol_idr / self.IDEAL_VOLUME_IDR) * 20)
                score += vol_score
                factors['volume'] = round(vol_score, 1)

                # 3. Buy pressure (max 20)
                pressure = min(20, (buy / sell) * 18)
                score += pressure
                factors['buy_pressure'] = round(pressure, 1)

                # 4. Spread quality (max 15)
                spread_q = max(0, (self.MAX_SPREAD_PCT - spread_pct) / self.MAX_SPREAD_PCT * 15)
                score += spread_q
                factors['spread'] = round(spread_q, 1)

                # 5. Trend position (max 20)
                if position_in_range > 0.85:
                    trend = 20
                elif position_in_range > 0.70:
                    trend = 15
                elif position_in_range > 0.55:
                    trend = 10
                elif position_in_range > 0.40:
                    trend = 5
                else:
                    trend = 1
                score += trend
                factors['trend'] = trend

                # 6. 24h change (max 10)
                mid = (high + low) / 2
                change_pct = ((last - mid) / mid) * 100
                if change_pct > 5:
                    cb = 10
                elif change_pct > 2:
                    cb = 7
                elif change_pct > 0:
                    cb = 4
                elif change_pct > -1:
                    cb = 1
                else:
                    cb = 0
                score += cb
                factors['24h_change'] = cb

                # ========== TIER 2: VELOCITY TRACKING ==========
                prev_score = self._previous_scores.get(clean_pair, score)
                velocity = score - prev_score  # Positive = improving
                self._previous_scores[clean_pair] = score
                factors['velocity'] = round(velocity, 1)

                # Micro-trend (price change since last scan)
                prev_price = self._previous_prices.get(clean_pair)
                micro_trend = 0
                if prev_price and prev_price > 0:
                    micro_trend = ((last - prev_price) / prev_price) * 100
                    if micro_trend > 0.3:
                        score += 5
                        factors['micro'] = 5
                    elif micro_trend < -0.5:
                        score -= 8
                        factors['micro'] = -8
                self._previous_prices[clean_pair] = last

                coin_name = clean_pair.replace('idr', '').upper()
                current_pairs.add(clean_pair)

                # Watchlist promotion check
                is_watchlisted = clean_pair in self._watchlist
                tier = 'scan'
                if is_watchlisted:
                    wl = self._watchlist[clean_pair]
                    wl['consecutive_hits'] += 1
                    wl['consecutive_misses'] = 0
                    wl['last_score'] = score
                    wl['velocity'] = velocity
                    tier = 'watchlist'
                    if wl['consecutive_hits'] >= self.WATCHLIST_PROMOTE_AFTER:
                        if velocity >= self.VELOCITY_BULLISH or score > 60:
                            tier = 'hot'
                            wl['hot'] = True
                else:
                    # First appearance — add to watchlist tracker
                    self._watchlist[clean_pair] = {
                        'consecutive_hits': 1,
                        'consecutive_misses': 0,
                        'first_seen': datetime.utcnow().isoformat(),
                        'last_score': score,
                        'velocity': velocity,
                        'hot': False,
                    }

                candidates.append({
                    'pair': clean_pair,
                    'coin': coin_name,
                    'price': last,
                    'high_24h': high,
                    'low_24h': low,
                    'bid': buy,
                    'ask': sell,
                    'volume_idr': vol_idr,
                    'spread_pct': round(spread_pct, 3),
                    'position_in_range': round(position_in_range, 3),
                    'price_change_pct': round(change_pct, 2),
                    'range_pct': round(range_pct, 2),
                    'score': round(score, 1),
                    'velocity': round(velocity, 1),
                    'micro_trend': round(micro_trend, 3),
                    'tier': tier,
                    'factors': factors,
                    'scanned_at': datetime.utcnow().isoformat(),
                })

                stats['passed'] += 1

            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                stats['rejected'] += 1
                continue

        # Demote watchlist coins that disappeared
        for pair in list(self._watchlist.keys()):
            if pair not in current_pairs:
                wl = self._watchlist[pair]
                wl['consecutive_misses'] += 1
                wl['consecutive_hits'] = 0
                wl['hot'] = False
                if wl['consecutive_misses'] >= self.WATCHLIST_DEMOTE_AFTER:
                    del self._watchlist[pair]

        # Sort: prioritize hot > watchlist > scan, then by score
        tier_priority = {'hot': 0, 'watchlist': 1, 'scan': 2}
        candidates.sort(key=lambda x: (tier_priority.get(x['tier'], 9), -x['score']))

        self._last_results = candidates[:self.TOP_N]
        self._last_scan = now
        self._scan_count += 1

        elapsed = round((time.time() - t0) * 1000)
        hot_count = sum(1 for c in candidates if c['tier'] == 'hot')
        wl_count = sum(1 for c in candidates if c['tier'] == 'watchlist')
        print(f"[Scanner] #{self._scan_count} | {stats['total']} pairs → {stats['passed']} passed "
              f"| Hot: {hot_count} | Watchlist: {wl_count} | {elapsed}ms")

        return self._last_results

    # ==========================================
    #  PUBLIC API
    # ==========================================

    def get_top_coin(self):
        results = self.scan_market()
        return results[0] if results else None

    def get_top_coins(self, n=5):
        results = self.scan_market()
        return results[:n]

    def get_hot_coins(self):
        """Get only Tier-3 (hot) coins ready for deep analysis."""
        results = self.scan_market()
        return [c for c in results if c['tier'] == 'hot']

    def get_watchlist_coins(self):
        """Get Tier-2 (watchlist) coins being tracked."""
        results = self.scan_market()
        return [c for c in results if c['tier'] in ('hot', 'watchlist')]

    def get_watchlist_summary(self):
        """Watchlist status for Telegram."""
        wl = self._watchlist
        if not wl:
            return "📋 Watchlist kosong — belum ada koin yang muncul konsisten."

        lines = ["📋 *Watchlist Status*", "━━━━━━━━━━━━━━━━━━━━━━"]
        for pair, info in sorted(wl.items(), key=lambda x: -x[1].get('last_score', 0)):
            hot = "🔥" if info.get('hot') else "👁️"
            coin = pair.replace('idr', '').upper()
            score = info.get('last_score', 0)
            vel = info.get('velocity', 0)
            hits = info.get('consecutive_hits', 0)
            vel_emoji = "📈" if vel > 0 else ("📉" if vel < 0 else "➡️")
            lines.append(
                f"{hot} *{coin}* | Score: {score:.0f} | {vel_emoji} Vel: {vel:+.1f} | Hits: {hits}x"
            )
        return "\n".join(lines)

    def format_scan_report(self, top_n=5):
        coins = self.get_top_coins(top_n)
        if not coins:
            return "📡 Tidak ada koin yang memenuhi filter saat ini."

        lines = ["🔍 *Market Scan — Top Coins*", "━━━━━━━━━━━━━━━━━━━━━━"]
        tier_emoji = {'hot': '🔥', 'watchlist': '👁️', 'scan': '📡'}
        for i, c in enumerate(coins, 1):
            te = tier_emoji.get(c['tier'], '📡')
            trend = "📈" if c['price_change_pct'] > 0 else "📉"
            vel = f" vel:{c['velocity']:+.0f}" if c['velocity'] != 0 else ""
            lines.append(
                f"{i}. {te} *{c['coin']}* — Score: {c['score']:.0f}/110{vel}\n"
                f"   💰 Rp {c['price']:,.0f} | {trend} {c['price_change_pct']:+.1f}%\n"
                f"   📊 Vol: Rp {c['volume_idr']/1e6:,.0f}M | Spread: {c['spread_pct']:.2f}%"
            )
        return "\n".join(lines)
