from flask import Flask, jsonify, request, session, redirect, url_for, render_template_string
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import time, secrets
from config import Config
from api import IndodaxAPI

app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET
limiter = Limiter(get_remote_address, app=app, default_limits=["60/minute"])
api = IndodaxAPI(Config.API_KEY, Config.SECRET_KEY)

def auth_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get('auth'):
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapper

LOGIN_HTML = '''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ISLM Monitor - Login</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Arial,sans-serif;background:#111;color:#fff;min-height:100vh;display:flex;align-items:center;justify-content:center}
.box{background:#1a1a1a;padding:40px;border-radius:16px;width:360px;border:1px solid #333}
h1{text-align:center;color:#4ade80;margin-bottom:25px}
input{width:100%;padding:14px;margin-bottom:15px;border:1px solid #333;border-radius:8px;background:#222;color:#fff;font-size:16px}
input:focus{outline:none;border-color:#4ade80}
button{width:100%;padding:14px;background:#4ade80;color:#000;border:none;border-radius:8px;font-size:16px;font-weight:bold;cursor:pointer}
button:hover{background:#22c55e}
.err{color:#f44;text-align:center;margin-bottom:15px}
.icon{text-align:center;font-size:50px;margin-bottom:15px}
</style></head>
<body><div class="box">
<div class="icon">🔐</div>
<h1>ISLM Monitor</h1>
{% if error %}<p class="err">{{ error }}</p>{% endif %}
<form method="POST">
<input type="password" name="pw" placeholder="Password" required autofocus>
<button>Login</button>
</form>
</div></body></html>'''

DASHBOARD_HTML = '''<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ISLM Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:Arial,sans-serif;background:#0d1117;color:#fff;padding:20px}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:20px;padding-bottom:15px;border-bottom:1px solid #333}
h1{color:#4ade80;font-size:22px}
.btns{display:flex;gap:10px}
.btn{background:#333;color:#fff;border:none;padding:10px 18px;border-radius:8px;cursor:pointer;text-decoration:none;font-size:14px}
.btn:hover{background:#444}
.btn-g{background:#4ade80;color:#000}
.live{display:flex;align-items:center;gap:8px;padding:8px 14px;background:rgba(74,222,128,0.1);border-radius:20px}
.dot{width:8px;height:8px;background:#4ade80;border-radius:50%;animation:p 2s infinite}
@keyframes p{0%,100%{opacity:1}50%{opacity:.5}}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin-bottom:20px}
@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:500px){.grid{grid-template-columns:1fr}}
.card{background:#161b22;padding:20px;border-radius:12px;border:1px solid #333}
.card-t{color:#888;font-size:12px;margin-bottom:8px;text-transform:uppercase}
.card-v{font-size:26px;font-weight:bold}
.card-s{color:#666;font-size:12px;margin-top:5px}
.pos{color:#4ade80}.neg{color:#f44}
.chart-box{background:#161b22;border-radius:12px;border:1px solid #333;padding:20px;margin-bottom:15px}
.chart-h{display:flex;justify-content:space-between;margin-bottom:15px}
.status{text-align:center;color:#666;font-size:11px}
</style></head>
<body>
<div class="header">
<h1>📊 ISLM Monitor</h1>
<div class="btns">
<div class="live"><span class="dot"></span><span style="color:#4ade80;font-size:12px">LIVE</span></div>
<button class="btn btn-g" onclick="load()">🔄 Refresh</button>
<a href="/logout" class="btn">Logout</a>
</div>
</div>
<div class="grid">
<div class="card"><div class="card-t">💰 ISLM Balance</div><div class="card-v" id="bal">--</div><div class="card-s" id="hold">Hold: --</div></div>
<div class="card"><div class="card-t">💹 Price</div><div class="card-v" id="price">--</div><div class="card-s" id="hl">H/L: --</div></div>
<div class="card"><div class="card-t">💎 Portfolio</div><div class="card-v" id="port">--</div><div class="card-s" id="idr">IDR: --</div></div>
<div class="card"><div class="card-t">📈 24H Volume</div><div class="card-v" id="vol">--</div><div class="card-s">ISLM</div></div>
</div>
<div class="chart-box">
<div class="chart-h"><span>📈 ISLM/IDR Candlestick</span><span id="upd" style="color:#666;font-size:12px">--</span></div>
<div id="chart"></div>
</div>
<p class="status">Last: <span id="last">--</span> | Auto-refresh: 10s</p>
<script>
let chart,data=[],islm=0,lp=0;
const fmt=n=>new Intl.NumberFormat('id-ID').format(n);
const rp=n=>'Rp '+fmt(Math.round(n));

function init(){
chart=new ApexCharts(document.getElementById('chart'),{
series:[{data:[]}],
chart:{type:'candlestick',height:350,background:'transparent',toolbar:{show:true}},
theme:{mode:'dark'},
xaxis:{type:'datetime',labels:{style:{colors:'#666'}}},
yaxis:{labels:{style:{colors:'#666'},formatter:v=>'Rp '+v}},
grid:{borderColor:'#333'},
plotOptions:{candlestick:{colors:{upward:'#4ade80',downward:'#f44'}}}
});
chart.render();
}

function upChart(p){
const now=Date.now(),t=Math.floor(now/60000)*60000;
let c=data.find(x=>x.x===t);
if(!c){c={x:t,y:[p,p,p,p]};data.push(c);if(data.length>30)data.shift();}
else{c.y[1]=Math.max(c.y[1],p);c.y[2]=Math.min(c.y[2],p);c.y[3]=p;}
chart.updateSeries([{data:data}]);
}

async function load(){
try{
const b=await(await fetch('/api/balance')).json();
if(b.success){islm=b.islm+b.islm_hold;document.getElementById('bal').textContent=fmt(islm.toFixed(2))+' ISLM';document.getElementById('hold').textContent='Hold: '+fmt(b.islm_hold.toFixed(2));document.getElementById('idr').textContent='IDR: '+rp(b.idr);}
else{document.getElementById('bal').innerHTML='<span class="neg">Error</span>';}

const p=await(await fetch('/api/price')).json();
if(p.success){
document.getElementById('price').textContent=rp(p.last);
document.getElementById('hl').textContent='H: '+rp(p.high)+' | L: '+rp(p.low);
document.getElementById('vol').textContent=fmt(Math.round(p.vol));
upChart(p.last);
if(islm>0)document.getElementById('port').textContent=rp(islm*p.last);
if(lp>0){const ch=((p.last-lp)/lp*100).toFixed(2);document.getElementById('price').innerHTML=rp(p.last)+(ch>=0?' <span class="pos">+'+ch+'%</span>':' <span class="neg">'+ch+'%</span>');}
lp=p.last;
}
document.getElementById('last').textContent=new Date().toLocaleTimeString('id-ID');
document.getElementById('upd').textContent='Updated: '+new Date().toLocaleTimeString('id-ID');
}catch(e){console.error(e);}
}

init();load();setInterval(load,10000);
</script>
</body></html>'''

@app.route('/')
def index():
    if session.get('auth'): return render_template_string(DASHBOARD_HTML)
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
@limiter.limit("5/minute")
def login():
    err=None
    if request.method=='POST':
        if request.form.get('pw')==Config.PASSWORD:
            session['auth']=True
            return redirect(url_for('index'))
        err="Invalid password"
        time.sleep(1)
    return render_template_string(LOGIN_HTML, error=err)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/api/balance')
@auth_required
def balance():
    return jsonify(api.get_balance())

@app.route('/api/price')
def price():
    return jsonify(api.get_price())

@app.after_request
def headers(r):
    r.headers['X-Content-Type-Options']='nosniff'
    r.headers['X-Frame-Options']='DENY'
    r.headers['Cache-Control']='no-store'
    return r

if __name__=='__main__':
    print("\\n=== ISLM Monitor ===")
    print(f"Server: http://{Config.HOST}:{Config.PORT}")
    print(f"Password: {Config.PASSWORD}\\n")
    app.run(host=Config.HOST, port=Config.PORT, debug=False)
