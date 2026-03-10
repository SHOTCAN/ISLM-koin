"""
Coin Scanner V3 — Fast Pre-Filter + Smart Ranking
===================================================
2-Stage scan for maximum speed:

  Stage 1: INSTANT PRE-FILTER (single API call, ~200ms)
    - get_summaries() → all pairs at once
    - Reject: dead coins, dust, no volume, inactive, falling
    - ~300 pairs filtered to ~15-30 candidates in milliseconds

  Stage 2: SCORING + RANKING (zero extra API calls)
    - Score remaining candidates on 6 factors
    - Micro-trend tracking between scans
    - Return top N sorted by score

The deep AI analysis (get_kline, multi-TF, etc.) happens ONLY
for the top 3 coins in the trading engine — NOT here.
"""

import time
from datetime import datetime


# Coins known to be stablecoins or wrapped assets (not worth trading)
BLACKLIST = {
    'usdtidr', 'usdcidr', 'busdidr', 'daiusd', 'tusdidr',
    'bidr', 'idrt', 'idrtidr',
}


class CoinScanner:
    """Fast 2-stage market scanner."""

    # === Stage 1: Pre-Filter Thresholds ===
    MIN_VOLUME_IDR = 5_000_000    # Rp5M (lowered to catch more candidates)
    MIN_PRICE_IDR = 10            # Skip dust
    MAX_SPREAD_PCT = 2.5          # Allow wider spread in pre-filter
    MIN_PRICE_RANGE_PCT = 0.5     # Min 24h range as % of price (skip flat coins)

    # === Stage 2: Scoring ===
    IDEAL_VOLUME_IDR = 50_000_000  # Rp50M = full volume score
    TOP_N = 10

    def __init__(self, api):
        self.api = api
        self._last_scan = None
        self._last_results = []
        self._scan_cooldown = 30   # Scan every 30s (fast enough now)
        self._previous_prices = {}
        self._scan_count = 0

    def scan_market(self):
        """
        2-stage scan: pre-filter → score → rank.
        Uses ONLY 1 API call (get_summaries). Zero kline calls.
        Returns sorted list of candidate dicts.
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
        rejected = {'no_idr': 0, 'blacklist': 0, 'dust': 0, 'no_vol': 0,
                     'no_bid_ask': 0, 'wide_spread': 0, 'flat': 0, 'falling': 0}

        for pair_id, data in tickers.items():
            try:
                # ========== STAGE 1: INSTANT PRE-FILTER ==========

                # IDR pairs only
                clean_pair = pair_id.replace('_', '').lower()
                if not clean_pair.endswith('idr'):
                    rejected['no_idr'] += 1
                    continue

                # Blacklist (stablecoins, wrapped tokens)
                if clean_pair in BLACKLIST:
                    rejected['blacklist'] += 1
                    continue

                last = float(data.get('last', 0))
                high = float(data.get('high', 0))
                low = float(data.get('low', 0))
                buy = float(data.get('buy', 0))
                sell = float(data.get('sell', 0))

                # Volume: try multiple field names (Indodax API inconsistency)
                vol_idr = 0
                for vk in ['vol_idr', 'vol_base', 'volume']:
                    v = data.get(vk)
                    if v:
                        try:
                            vol_idr = float(v)
                            if vol_idr > 0:
                                break
                        except (ValueError, TypeError):
                            continue

                # Filter 1: Dust coins
                if last < self.MIN_PRICE_IDR:
                    rejected['dust'] += 1
                    continue

                # Filter 2: Dead volume
                if vol_idr < self.MIN_VOLUME_IDR:
                    rejected['no_vol'] += 1
                    continue

                # Filter 3: No bid/ask (inactive order book)
                if buy <= 0 or sell <= 0 or sell <= buy:
                    rejected['no_bid_ask'] += 1
                    continue

                # Filter 4: Wide spread
                spread_pct = ((sell - buy) / buy) * 100
                if spread_pct > self.MAX_SPREAD_PCT:
                    rejected['wide_spread'] += 1
                    continue

                # Filter 5: Flat coin (no movement)
                if high <= low or high == 0:
                    rejected['flat'] += 1
                    continue
                price_range = high - low
                range_pct = (price_range / low) * 100
                if range_pct < self.MIN_PRICE_RANGE_PCT:
                    rejected['flat'] += 1
                    continue

                # Filter 6: Position in 24h range (reject bottom 25%)
                position_in_range = (last - low) / price_range
                if position_in_range < 0.25:
                    rejected['falling'] += 1
                    continue

                # ========== STAGE 2: SCORING (6 factors, 0-110) ==========
                score = 0
                factors = {}

                # 1. Momentum: position in 24h range (max 25)
                momentum = position_in_range * 25
                score += momentum
                factors['momentum'] = round(momentum, 1)

                # 2. Volume strength (max 20)
                vol_score = min(20, (vol_idr / self.IDEAL_VOLUME_IDR) * 20)
                score += vol_score
                factors['volume'] = round(vol_score, 1)

                # 3. Buy pressure (max 20)
                buy_ratio = buy / sell
                pressure = min(20, buy_ratio * 18)
                score += pressure
                factors['buy_pressure'] = round(pressure, 1)

                # 4. Spread quality (max 15)
                spread_quality = max(0, (self.MAX_SPREAD_PCT - spread_pct) / self.MAX_SPREAD_PCT * 15)
                score += spread_quality
                factors['spread'] = round(spread_quality, 1)

                # 5. Trend position (max 20)
                if position_in_range > 0.85:
                    trend = 20   # Near daily high
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

                # 6. 24h change estimate (max 10)
                mid_price = (high + low) / 2
                change_pct = ((last - mid_price) / mid_price) * 100
                if change_pct > 5:
                    change_bonus = 10
                elif change_pct > 2:
                    change_bonus = 7
                elif change_pct > 0:
                    change_bonus = 4
                elif change_pct > -1:
                    change_bonus = 1
                else:
                    change_bonus = 0
                score += change_bonus
                factors['24h_change'] = change_bonus

                # Micro-trend bonus/penalty (between scans)
                prev = self._previous_prices.get(clean_pair)
                if prev and prev > 0:
                    micro = ((last - prev) / prev) * 100
                    if micro > 0.3:
                        score += 5
                        factors['micro_trend'] = 5
                    elif micro < -0.5:
                        score -= 8
                        factors['micro_trend'] = -8
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
                    'price_change_pct': round(change_pct, 2),
                    'range_pct': round(range_pct, 2),
                    'score': round(score, 1),
                    'factors': factors,
                    'scanned_at': datetime.utcnow().isoformat(),
                })

            except (ValueError, TypeError, KeyError, ZeroDivisionError):
                continue

        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)
        self._last_results = candidates[:self.TOP_N]
        self._last_scan = now
        self._scan_count += 1

        elapsed = round((time.time() - t0) * 1000)
        total = len(tickers)
        passed = len(candidates)
        print(f"[Scanner] #{self._scan_count} | {total} pairs → {passed} passed → Top {len(self._last_results)} | {elapsed}ms | "
              f"Rejected: vol={rejected['no_vol']} spread={rejected['wide_spread']} flat={rejected['flat']} fall={rejected['falling']}")

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
                f"{medal} *{c['coin']}* — Score: {c['score']}/110\n"
                f"   💰 Rp {c['price']:,.0f} | {trend_emoji} {c['price_change_pct']:+.1f}%\n"
                f"   📊 Vol: Rp {c['volume_idr']/1e6:,.0f}M | Spread: {c['spread_pct']:.2f}% | Range: {c['range_pct']:.1f}%"
            )
        return "\n".join(lines)
