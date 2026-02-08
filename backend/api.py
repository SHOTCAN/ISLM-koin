import hmac, hashlib, time, urllib.parse, requests

class IndodaxAPI:
    def __init__(self, api_key, secret_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.nonce = int(time.time() * 1000)
    
    def _sign(self, params):
        query = urllib.parse.urlencode(params)
        return hmac.new(self.secret_key.encode(), query.encode(), hashlib.sha512).hexdigest()
    
    def _private(self, method, params=None):
        if params is None: params = {}
        params['method'] = method
        self.nonce += 1
        params['nonce'] = self.nonce
        try:
            r = requests.post('https://indodax.com/tapi', 
                headers={'Key': self.api_key, 'Sign': self._sign(params)}, 
                data=params, timeout=10)
            return r.json()
        except Exception as e:
            return {'success': 0, 'error': str(e)}
    
    def get_balance(self):
        info = self._private('getInfo')
        if info.get('success') == 1:
            bal = info.get('return', {}).get('balance', {})
            hold = info.get('return', {}).get('balance_hold', {})
            return {
                'success': True,
                'islm': float(bal.get('islm', 0)),
                'islm_hold': float(hold.get('islm', 0)),
                'idr': float(bal.get('idr', 0))
            }
        return {'success': False, 'error': info.get('error', 'Error')}
    
    def get_price(self, pair='islmidr'):
        try:
            r = requests.get(f'https://indodax.com/api/ticker/{pair}', timeout=10)
            t = r.json().get('ticker', {})
            return {
                'success': True,
                'last': float(t.get('last', 0)),
                'high': float(t.get('high', 0)),
                'low': float(t.get('low', 0)),
                'vol': float(t.get('vol_' + pair.replace('idr', ''), 0))
            }
        except:
            return {'success': False}

    def get_depth(self, pair='islmidr'):
        try:
            r = requests.get(f'https://indodax.com/api/depth/{pair}', timeout=10)
            return r.json() # Returns {'buy': [], 'sell': []}
        except:
            return {'buy': [], 'sell': []}
