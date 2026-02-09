# 🚀 DEPLOY BOT GRATIS 24/7 — Step by Step

## Option A: Koyeb (GRATIS, Recommended ⭐)

Koyeb punya free tier yang bisa jalankan bot Python 24/7.

### Langkah-langkah:

**1. Daftar Koyeb**
- Buka [koyeb.com](https://www.koyeb.com)
- Sign up pakai GitHub (gratis)

**2. Buat Service Baru**
- Klik **"Create Service"**
- Pilih **"GitHub"** sebagai source
- Pilih repository: `SHOTCAN/ISLM-koin`
- Branch: `main`

**3. Konfigurasi**
- Service type: **Worker** (bukan Web!)
- Builder: **Dockerfile**
- Instance type: **Free / Nano**
- Region: **Closest to you**

**4. Set Environment Variables**
Klik "Environment Variables" dan tambahkan:
```
INDODAX_API_KEY     = (isi API key kamu)
INDODAX_SECRET_KEY  = (isi secret key kamu)
TELEGRAM_TOKEN      = (isi bot token)
TELEGRAM_CHAT_ID    = (isi chat ID)
```

**5. Deploy!**
- Klik **"Deploy"**
- Tunggu build selesai (~2 menit)
- Bot langsung aktif 24/7! ✅

---

## Option B: Railway (Gratis $5/bulan credit)

Railway kasih $5 free credit per bulan — cukup untuk bot kecil.

### Langkah-langkah:

**1. Daftar Railway**
- Buka [railway.app](https://railway.app)
- Sign up pakai GitHub

**2. New Project**
- Klik **"New Project"** → **"Deploy from GitHub repo"**
- Pilih `SHOTCAN/ISLM-koin`

**3. Set Environment Variables**
Di tab "Variables", tambahkan:
```
INDODAX_API_KEY     = (isi)
INDODAX_SECRET_KEY  = (isi)
TELEGRAM_TOKEN      = (isi)
TELEGRAM_CHAT_ID    = (isi)
```

**4. Set Start Command**
Di tab "Settings" → Custom Start Command:
```
python bot_standalone.py
```

**5. Deploy!**
- Railway otomatis deploy dari GitHub
- Bot aktif 24/7 selama credit tersisa ✅

---

## Option C: Render (Gratis tapi ada limit)

> ⚠️ Free tier Render meng-sleep service setelah 15 menit tanpa request.
> Untuk bot Telegram (yang polling), ini BISA jalan karena bot terus polling.
> Tapi tidak 100% reliable.

### Langkah:
1. Buka [render.com](https://render.com)
2. New → Background Worker
3. Connect GitHub repo
4. Set env vars
5. Start command: `python bot_standalone.py`

---

## Tabel Perbandingan

| Platform | Gratis? | 24/7? | Setup |
|----------|---------|-------|-------|
| **Koyeb** | ✅ Ya | ✅ Ya | Mudah |
| **Railway** | ✅ $5/bulan | ✅ Ya | Mudah |
| **Render** | ✅ Ya | ⚠️ Kadang sleep | Mudah |
| **Laptop** | ✅ | ❌ Mati = OFF | Instant |

---

## Setelah Deploy

1. Bot kirim pesan: _"🟢 ISLM Bot Standalone AKTIF"_
2. Auto-update setiap 5 menit
3. Ketik `/menu` di Telegram untuk cek
4. Selesai! 🎉
