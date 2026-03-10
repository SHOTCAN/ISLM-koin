"""
Coin Scanner V2 — Ultra-Safe Market Intelligence Engine
========================================================
Scans ALL Indodax trading pairs with STRICT filters:
  1. Volume filter (only liquid coins > Rp20M/day)
  2. Spread filter (only tight-spread coins < 1.5%)
  3. Price change filter (REJECT falling coins)
  4. Momentum scoring (position in 24h range)
  5. Buy pressure analysis (bid vs ask)
  6. Trend validation (must be in uptrend)

Only returns coins with STRONG bullish probability.
"""

import time
from datetime import datetime


class CoinScanner:
    """Ultra-safe market scanner — only pick the best coins."""

    # Minimum 24h volume in IDR
    MIN_VOLUME_IDR = 20_000_000  # Rp20M minimum (liquid only)
    # Minimum price
    MIN_PRICE_IDR = 100  # Skip micro-coins
    # Maximum spread
    MAX_SPREAD_PCT = 1.5
    # Max coins to return
    TOP_N = 8

    def __init__(self, api):
        self.api = api
        self._last_scan = None
        self._last_results = []
        self._scan_cooldown = 45  # 45s between scans
        self._previous_prices = {}  # pair -> previous price for trend

    def scan_market(self):
        """
        Full market scan with ultra-strict filters.
        Returns list of coin dicts sorted by score (highest first).
        """
        now = time.time()
        if self._last_scan and (now - self._last_scan) < self._scan_cooldown:
            return self._last_results

        summaries = self.api.get_summaries()
        if not summaries.get('success'):
            print(f"[Scanner] Failed to fetch summaries: {summaries.get('error')}")
            return self._last_results

        tickers = summaries.get('tickers', {})
        candidates = []

        for pair_id, data in tickers.items():
            try:
                # Only IDR pairs
                if not pair_id.endswith('idr') and '_idr' not in pair_id:
                    continue

                last = float(data.get('last', 0))
                high = float(data.get('high', 0))
                low = float(data.get('low', 0))
                buy = float(data.get('buy', 0))
                sell = float(data.get('sell', 0))
                vol_idr = float(data.get('vol_idr', 0) or data.get('vol_base', 0) or 0)

                # ===== STRICT FILTERS =====
                if last < self.MIN_PRICE_IDR:
                    continue
                if vol_idr < self.MIN_VOLUME_IDR:
                    continue

                # Spread check
                spread_pct = 0
                if buy > 0 and sell > 0:
                    spread_pct = ((sell - buy) / buy) * 100
                    if spread_pct > self.MAX_SPREAD_PCT:
                        continue
                else:
                    continue  # No bid/ask = skip

                # 24h range check — REJECT coins where high == low (no movement)
                if high <= low or high == 0:
                    continue

                price_range = high - low
                position_in_range = (last - low) / price_range

                # ===== CRITICAL: REJECT FALLING COINS =====
                # If price is in bottom 30% of 24h range → coin is falling, SKIP
                if position_in_range < 0.30:
                    continue

                # Calculate 24h price change estimate
                mid_price = (high + low) / 2
                price_change_pct = ((last - mid_price) / mid_price) * 100

                # REJECT negative 24h change (coin trending down)
                if price_change_pct < -1.0:
                    continue

                # ===== SCORING (0-100) =====
                score = 0
                factors = {}

                # 1. Momentum: position in 24h range (max 25 pts)
                # Higher = more bullish (price near high)
                momentum_score = position_in_range * 25
                score += momentum_score
                factors['momentum'] = round(momentum_score, 1)

                # 2. Volume strength (max 20 pts)
                vol_score = min(20, (vol_idr / self.MIN_VOLUME_IDR) * 5)
                score += vol_score
                factors['volume'] = round(vol_score, 1)

                # 3. Buy pressure: bid close to ask = strong demand (max 20 pts)
                if buy > 0 and sell > 0:
                    buy_ratio = buy / sell  # Closer to 1 = strong buyers
                    pressure_score = min(20, buy_ratio * 15)
                    score += pressure_score
                    factors['buy_pressure'] = round(pressure_score, 1)

                # 4. Spread quality: tighter = better (max 15 pts)
                spread_score = max(0, (self.MAX_SPREAD_PCT - spread_pct) / self.MAX_SPREAD_PCT * 15)
                score += spread_score
                factors['spread'] = round(spread_score, 1)

                # 5. Trend strength: near daily high = breakout potential (max 20 pts)
                if position_in_range > 0.80:
                    trend_score = 20  # Very close to high = strong uptrend
                elif position_in_range > 0.65:
                    trend_score = 14
                elif position_in_range > 0.50:
                    trend_score = 8
                else:
                    trend_score = 2
                score += trend_score
                factors['trend'] = round(trend_score, 1)

                # 6. BONUS: Positive 24h change (max 10 pts)
                if price_change_pct > 3:
                    change_bonus = 10
                elif price_change_pct > 1:
                    change_bonus = 6
                elif price_change_pct > 0:
                    change_bonus = 3
                else:
                    change_bonus = 0
                score += change_bonus
                factors['24h_change'] = round(change_bonus, 1)

                # Track previous prices for micro-trend
                clean_pair = pair_id.replace('_', '')
                prev = self._previous_prices.get(clean_pair, last)
                if prev > 0:
                    micro_trend = ((last - prev) / prev) * 100
                    # REJECT if micro-trend is strongly negative
                    if micro_trend < -0.5:
                        score -= 10
                        factors['micro_trend'] = -10
                    elif micro_trend > 0.2:
                        score += 5
                        factors['micro_trend'] = 5
                self._previous_prices[clean_pair] = last

                coin_name = clean_pair.replace('idr', '').upper()

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
                    'price_change_pct': round(price_change_pct, 2),
                    'score': round(score, 1),
                    'factors': factors,
                    'scanned_at': datetime.utcnow().isoformat(),
                })

            except (ValueError, TypeError, KeyError):
                continue

        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        self._last_results = candidates[:self.TOP_N]
        self._last_scan = now

        passed = len(candidates)
        total = len(tickers)
        print(f"[Scanner] {total} pairs → {passed} passed → Top {len(self._last_results)}")
        return self._last_results

    def get_top_coin(self):
        results = self.scan_market()
        return results[0] if results else None

    def get_top_coins(self, n=5):
        results = self.scan_market()
        return results[:n]

    def format_scan_report(self, top_n=5):
        coins = self.get_top_coins(top_n)
        if not coins:
            return "📡 Tidak ada koin yang memenuhi filter saat ini."

        lines = ["🔍 *Market Scan — Top Coins*", "━━━━━━━━━━━━━━━━━━━━━━"]
        medals = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣']
        for i, c in enumerate(coins, 1):
            medal = medals[i-1] if i <= 5 else f'{i}.'
            trend_emoji = "📈" if c['price_change_pct'] > 0 else "📉"
            lines.append(
                f"{medal} *{c['coin']}* — Score: {c['score']}/100\n"
                f"   💰 Rp {c['price']:,.0f} | {trend_emoji} {c['price_change_pct']:+.1f}%\n"
                f"   📊 Vol: Rp {c['volume_idr']/1e6:,.0f}M | Spread: {c['spread_pct']:.2f}%"
            )
        return "\n".join(lines)
