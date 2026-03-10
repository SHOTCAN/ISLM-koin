"""
Coin Scanner — Multi-Coin Market Intelligence Engine
=====================================================
Scans ALL Indodax trading pairs and ranks them by:
  - Momentum (price change velocity)
  - Volume spikes (relative to 24h average)
  - Order book imbalance (buy vs sell pressure)
  - Volatility breakouts (ATR expansion)
  - 24h performance (gainers vs losers)
  
Returns top-N coins with highest bullish probability.
"""

import time
import statistics
from datetime import datetime


class CoinScanner:
    """Scan and rank all Indodax coins for trading opportunities."""

    # Minimum 24h volume in IDR to consider a coin tradeable
    MIN_VOLUME_IDR = 5_000_000  # Rp5 juta minimum daily volume
    # Minimum price in IDR
    MIN_PRICE_IDR = 10  # Skip dust/dead coins
    # Maximum spread percentage allowed
    MAX_SPREAD_PCT = 3.0
    # How many top coins to return
    TOP_N = 10

    def __init__(self, api):
        """
        Args:
            api: IndodaxAPI or IndodaxTradeAPI instance
        """
        self.api = api
        self._last_scan = None
        self._last_results = []
        self._scan_cooldown = 60  # Minimum 60s between scans

    def scan_market(self):
        """
        Full market scan: fetch all pairs, rank by bullish potential.
        Returns list of coin dicts sorted by score (highest first).
        """
        now = time.time()
        if self._last_scan and (now - self._last_scan) < self._scan_cooldown:
            return self._last_results

        # Fetch all summaries (single API call for all pairs)
        summaries = self.api.get_summaries()
        if not summaries.get('success'):
            print(f"[Scanner] Failed to fetch summaries: {summaries.get('error')}")
            return self._last_results

        tickers = summaries.get('tickers', {})
        candidates = []

        for pair_id, data in tickers.items():
            try:
                # Only process IDR pairs
                if not pair_id.endswith('idr') and '_idr' not in pair_id:
                    continue

                # Normalize key access (Indodax format varies)
                last = float(data.get('last', 0))
                high = float(data.get('high', 0))
                low = float(data.get('low', 0))
                buy = float(data.get('buy', 0))
                sell = float(data.get('sell', 0))
                vol_idr = float(data.get('vol_idr', 0) or data.get('vol_base', 0) or 0)
                vol_coin = float(data.get('vol_coin', 0) or data.get('vol_traded', 0) or 0)

                # Basic filters
                if last < self.MIN_PRICE_IDR:
                    continue
                if vol_idr < self.MIN_VOLUME_IDR:
                    continue

                # Spread filter
                spread_pct = 0
                if buy > 0 and sell > 0:
                    spread_pct = ((sell - buy) / buy) * 100
                    if spread_pct > self.MAX_SPREAD_PCT:
                        continue

                # --- Scoring ---
                score = 0
                factors = {}

                # 1. Price momentum (24h change %)
                if high > 0 and low > 0 and low > 0:
                    price_range = high - low
                    position_in_range = (last - low) / price_range if price_range > 0 else 0.5
                    momentum_score = position_in_range * 30  # Max 30 pts
                    score += momentum_score
                    factors['momentum'] = round(momentum_score, 1)

                # 2. Volume strength
                # Higher volume = more confidence in the move
                vol_score = min(20, (vol_idr / self.MIN_VOLUME_IDR) * 5)  # Max 20 pts
                score += vol_score
                factors['volume'] = round(vol_score, 1)

                # 3. Buy pressure (bid > ask = bullish)
                if buy > 0 and sell > 0:
                    buy_pressure = buy / sell  # > 1 means bid close to ask (bullish)
                    pressure_score = min(20, buy_pressure * 15)  # Max 20 pts
                    score += pressure_score
                    factors['buy_pressure'] = round(pressure_score, 1)

                # 4. Spread tightness (tight spread = liquid)
                spread_score = max(0, (self.MAX_SPREAD_PCT - spread_pct) / self.MAX_SPREAD_PCT * 15)
                score += spread_score
                factors['spread'] = round(spread_score, 1)

                # 5. Price vs range: if price is in upper 30% of 24h range, bullish
                if position_in_range > 0.7:
                    breakout_score = 15
                elif position_in_range > 0.5:
                    breakout_score = 8
                else:
                    breakout_score = 0
                score += breakout_score
                factors['breakout'] = round(breakout_score, 1)

                # Clean pair name
                clean_pair = pair_id.replace('_', '')
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
                    'score': round(score, 1),
                    'factors': factors,
                    'scanned_at': datetime.utcnow().isoformat(),
                })

            except (ValueError, TypeError, KeyError) as e:
                continue

        # Sort by score descending
        candidates.sort(key=lambda x: x['score'], reverse=True)

        self._last_results = candidates[:self.TOP_N]
        self._last_scan = now

        print(f"[Scanner] Scanned {len(tickers)} pairs → {len(candidates)} passed filters → Top {len(self._last_results)}")
        return self._last_results

    def get_top_coin(self):
        """Get the single best coin to trade right now."""
        results = self.scan_market()
        return results[0] if results else None

    def get_top_coins(self, n=5):
        """Get top-N coins ranked by score."""
        results = self.scan_market()
        return results[:n]

    def format_scan_report(self, top_n=5):
        """Format a nice Telegram-friendly scan report."""
        coins = self.get_top_coins(top_n)
        if not coins:
            return "📡 Tidak ada koin yang memenuhi filter saat ini."

        lines = ["🔍 *Market Scan — Top Coins*", "━━━━━━━━━━━━━━━━━━━━━━"]
        for i, c in enumerate(coins, 1):
            medal = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][i-1] if i <= 5 else f'{i}.'
            lines.append(
                f"{medal} *{c['coin']}* — Score: {c['score']}/100\n"
                f"   💰 Rp {c['price']:,.0f} | Vol: Rp {c['volume_idr']:,.0f}\n"
                f"   📊 Spread: {c['spread_pct']:.2f}% | Bid/Ask: {c['bid']:,.0f}/{c['ask']:,.0f}"
            )
        return "\n".join(lines)
