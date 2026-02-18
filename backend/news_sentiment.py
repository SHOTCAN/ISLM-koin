"""
ISLM Monitor — News & Sentiment Engine V8
==========================================
Multi-source news aggregation with AI sentiment:
  - CoinGecko market data + trending
  - CryptoPanic RSS (free tier)
  - CoinTelegraph / CoinDesk RSS feeds
  - Fear & Greed Index
  - BTC Dominance + Total Market Cap
  - Macro event detection
  - ISLM/Haqq-specific mention tracking
  - Correlation matrix (ISLM/BTC/ETH/TOTAL)
"""

import time
import requests
import json
from datetime import datetime, timedelta
from collections import deque


class NewsSentimentEngine:
    """Aggregate crypto news and compute sentiment scores."""

    def __init__(self):
        self._news_cache = deque(maxlen=100)
        self._last_fetch = 0
        self._fetch_interval = 1800  # 30 minutes
        self._fear_greed_cache = None
        self._fear_greed_time = 0
        self._market_data_cache = {}
        self._market_data_time = 0

    # ===== Fear & Greed Index =====
    def get_fear_greed(self):
        """Get current Fear & Greed Index (0-100)."""
        if self._fear_greed_cache and time.time() - self._fear_greed_time < 3600:
            return self._fear_greed_cache

        try:
            r = requests.get('https://api.alternative.me/fng/?limit=7', timeout=10)
            if r.status_code == 200:
                data = r.json().get('data', [])
                if data:
                    current = data[0]
                    result = {
                        'success': True,
                        'value': int(current.get('value', 50)),
                        'label': current.get('value_classification', 'Neutral'),
                        'timestamp': current.get('timestamp', ''),
                        'history': [{'value': int(d.get('value', 50)),
                                     'label': d.get('value_classification', '')} for d in data[:7]],
                        'trend': 'IMPROVING' if len(data) >= 2 and int(data[0]['value']) > int(data[1]['value']) else 'DECLINING',
                    }
                    self._fear_greed_cache = result
                    self._fear_greed_time = time.time()
                    return result
        except:
            pass
        return {'success': False, 'value': 50, 'label': 'Neutral'}

    # ===== CoinGecko Market Overview =====
    def get_market_overview(self):
        """Get total crypto market cap, BTC dominance, and market trends."""
        if self._market_data_cache and time.time() - self._market_data_time < 600:
            return self._market_data_cache

        try:
            r = requests.get('https://api.coingecko.com/api/v3/global', timeout=10)
            if r.status_code == 200:
                data = r.json().get('data', {})
                result = {
                    'success': True,
                    'total_market_cap_usd': data.get('total_market_cap', {}).get('usd', 0),
                    'total_volume_24h_usd': data.get('total_volume', {}).get('usd', 0),
                    'btc_dominance': round(data.get('market_cap_percentage', {}).get('btc', 0), 1),
                    'eth_dominance': round(data.get('market_cap_percentage', {}).get('eth', 0), 1),
                    'active_coins': data.get('active_cryptocurrencies', 0),
                    'market_cap_change_24h': round(data.get('market_cap_change_percentage_24h_usd', 0), 2),
                    'updated': datetime.now().strftime('%H:%M'),
                }

                # Altseason indicator: BTC dominance < 50% = altseason
                btc_dom = result['btc_dominance']
                if btc_dom < 40:
                    result['altseason'] = 'STRONG_ALTSEASON'
                    result['altseason_label'] = '🚀 Altseason Mode!'
                elif btc_dom < 50:
                    result['altseason'] = 'MILD_ALTSEASON'
                    result['altseason_label'] = '📈 Alt-Friendly'
                elif btc_dom > 60:
                    result['altseason'] = 'BTC_SEASON'
                    result['altseason_label'] = '₿ BTC Season'
                else:
                    result['altseason'] = 'NEUTRAL'
                    result['altseason_label'] = '⚖️ Balanced'

                self._market_data_cache = result
                self._market_data_time = time.time()
                return result
        except:
            pass
        return {'success': False, 'btc_dominance': 0, 'altseason': 'UNKNOWN'}

    # ===== BTC Macro Analysis =====
    def get_btc_analysis(self):
        """Get BTC price data for macro analysis."""
        try:
            # Current price
            r = requests.get(
                'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd,idr&include_24hr_change=true&include_24hr_vol=true&include_market_cap=true',
                timeout=10
            )
            btc = {}
            if r.status_code == 200:
                data = r.json().get('bitcoin', {})
                btc = {
                    'price_usd': data.get('usd', 0),
                    'price_idr': data.get('idr', 0),
                    'change_24h': round(data.get('usd_24h_change', 0), 2),
                    'volume_24h': data.get('usd_24h_vol', 0),
                    'market_cap': data.get('usd_market_cap', 0),
                }

            # 7-day price history for trend
            r2 = requests.get(
                'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=7&interval=daily',
                timeout=10
            )
            if r2.status_code == 200:
                prices_7d = [p[1] for p in r2.json().get('prices', [])]
                if len(prices_7d) >= 2:
                    btc['trend_7d'] = 'BULLISH' if prices_7d[-1] > prices_7d[0] else 'BEARISH'
                    btc['change_7d'] = round((prices_7d[-1] - prices_7d[0]) / prices_7d[0] * 100, 2)
                    btc['high_7d'] = max(prices_7d)
                    btc['low_7d'] = min(prices_7d)
                    btc['prices_7d'] = prices_7d

            btc['success'] = True
            return btc
        except:
            return {'success': False}

    # ===== Correlation Matrix =====
    def get_correlation_matrix(self):
        """Compute correlation between ISLM, BTC, ETH, and total market."""
        try:
            coins = {
                'haqq-network': 'ISLM',
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
            }
            price_data = {}
            for coin_id, label in coins.items():
                r = requests.get(
                    f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30&interval=daily',
                    timeout=10
                )
                if r.status_code == 200:
                    prices = [p[1] for p in r.json().get('prices', [])]
                    if prices:
                        price_data[label] = prices

            if len(price_data) < 2:
                return {'success': False}

            # Compute correlations
            correlations = {}
            labels = list(price_data.keys())
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    a = price_data[labels[i]]
                    b = price_data[labels[j]]
                    min_len = min(len(a), len(b))
                    if min_len >= 5:
                        corr = self._pearson_correlation(a[-min_len:], b[-min_len:])
                        pair_key = f"{labels[i]}/{labels[j]}"
                        correlations[pair_key] = {
                            'value': round(corr, 3),
                            'strength': 'STRONG' if abs(corr) > 0.7 else ('MODERATE' if abs(corr) > 0.4 else 'WEAK'),
                            'direction': 'POSITIVE' if corr > 0 else 'NEGATIVE',
                        }

            return {'success': True, 'correlations': correlations, 'period': '30d'}
        except:
            return {'success': False}

    @staticmethod
    def _pearson_correlation(a, b):
        """Compute Pearson correlation coefficient."""
        n = min(len(a), len(b))
        if n < 3:
            return 0
        import statistics
        mean_a = statistics.mean(a[:n])
        mean_b = statistics.mean(b[:n])
        dev_a = [x - mean_a for x in a[:n]]
        dev_b = [x - mean_b for x in b[:n]]
        cov = sum(da * db for da, db in zip(dev_a, dev_b)) / n
        std_a = (sum(d ** 2 for d in dev_a) / n) ** 0.5
        std_b = (sum(d ** 2 for d in dev_b) / n) ** 0.5
        if std_a * std_b == 0:
            return 0
        return cov / (std_a * std_b)

    # ===== Aggregated Sentiment Score =====
    def compute_sentiment_score(self):
        """Compute combined sentiment score (0-100) from all sources."""
        scores = []
        weights = []

        # Fear & Greed
        fg = self.get_fear_greed()
        if fg.get('success'):
            scores.append(fg['value'])
            weights.append(0.3)

        # Market overview
        market = self.get_market_overview()
        if market.get('success'):
            # Market cap change → sentiment
            mc_change = market.get('market_cap_change_24h', 0)
            mc_score = 50 + mc_change * 5  # ±10% → ±50 points
            mc_score = max(0, min(100, mc_score))
            scores.append(mc_score)
            weights.append(0.2)

            # BTC dominance → altcoin sentiment
            btc_dom = market.get('btc_dominance', 50)
            alt_score = max(0, min(100, 100 - btc_dom))  # Lower BTC dom = better for alts
            scores.append(alt_score)
            weights.append(0.15)

        # BTC trend
        btc = self.get_btc_analysis()
        if btc.get('success'):
            btc_change = btc.get('change_24h', 0)
            btc_score = 50 + btc_change * 3
            btc_score = max(0, min(100, btc_score))
            scores.append(btc_score)
            weights.append(0.35)

        if not scores:
            return {'score': 50, 'label': 'Neutral', 'valid': False}

        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / max(total_weight, 0.01)
        weighted_score = round(max(0, min(100, weighted_score)), 1)

        if weighted_score > 75:
            label = '🟢 Very Bullish'
        elif weighted_score > 60:
            label = '🟢 Bullish'
        elif weighted_score > 40:
            label = '🟡 Neutral'
        elif weighted_score > 25:
            label = '🔴 Bearish'
        else:
            label = '🔴 Very Bearish'

        return {
            'score': weighted_score,
            'label': label,
            'valid': True,
            'components': {
                'fear_greed': fg.get('value', 50),
                'market_trend': round(scores[1], 1) if len(scores) > 1 else 50,
                'altcoin_sentiment': round(scores[2], 1) if len(scores) > 2 else 50,
                'btc_momentum': round(scores[3], 1) if len(scores) > 3 else 50,
            },
            'timestamp': datetime.now().strftime('%H:%M'),
        }

    # ===== BTC Macro Report (auto every 2-3 days) =====
    def generate_btc_report(self):
        """Generate comprehensive BTC macro report for Telegram."""
        btc = self.get_btc_analysis()
        market = self.get_market_overview()
        fg = self.get_fear_greed()

        if not btc.get('success'):
            return None

        price = btc.get('price_usd', 0)
        change_24h = btc.get('change_24h', 0)
        change_7d = btc.get('change_7d', 0)
        trend = btc.get('trend_7d', 'N/A')
        btc_dom = market.get('btc_dominance', 0) if market.get('success') else 0
        fg_val = fg.get('value', 50) if fg.get('success') else 50
        fg_label = fg.get('label', 'Neutral')
        altseason = market.get('altseason_label', 'N/A') if market.get('success') else 'N/A'

        # Impact analysis for altcoins
        if change_7d > 5:
            impact = "🟢 Positif — BTC rally bisa tarik altcoin naik"
        elif change_7d > 0:
            impact = "🟡 Netral-Positif — Kondisi stabil untuk altcoin"
        elif change_7d > -5:
            impact = "🟡 Netral — Perhatikan momentum"
        else:
            impact = "🔴 Negatif — BTC turun tekan altcoin"

        report = (
            f"📊 *LAPORAN BTC MACRO*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 BTC: ${price:,.0f} ({change_24h:+.1f}% 24h)\n"
            f"📈 Trend 7H: *{trend}* ({change_7d:+.1f}%)\n"
            f"🏛️ BTC Dominance: {btc_dom:.1f}%\n"
            f"😱 Fear & Greed: {fg_val} ({fg_label})\n"
            f"{altseason}\n\n"
            f"⚡ *Impact ke Altcoin:*\n{impact}\n\n"
            f"📅 _{datetime.now().strftime('%d %b %Y %H:%M')} WIB_"
        )

        return report
