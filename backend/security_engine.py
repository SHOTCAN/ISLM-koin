"""
ISLM Monitor — AI Security Engine V5
=====================================
Multi-layer security system with real-time monitoring:
  - Rate Limiter (per IP/user)
  - Brute Force Shield (progressive lockout)
  - Session Guardian (hijack detection)
  - Anomaly Detector (unusual patterns)
  - Security Alert System (via Telegram)
"""

import time
import hashlib
import hmac
import json
import os
from datetime import datetime
from collections import defaultdict


class SecurityEngine:
    """AI-powered security monitoring system."""

    def __init__(self, telegram_token=None, chat_id=None):
        self.telegram_token = telegram_token
        self.chat_id = chat_id

        # Rate limiting
        self._request_log = defaultdict(list)  # ip -> [timestamps]
        self.RATE_LIMIT = 30  # max requests per minute
        self.RATE_WINDOW = 60  # seconds

        # Brute force
        self._failed_attempts = defaultdict(int)  # ip -> count
        self._lockout_until = defaultdict(float)  # ip -> timestamp
        self.LOCKOUT_STAGES = [300, 1800, 3600]  # 5min, 30min, 1hr

        # Session tracking
        self._active_sessions = {}  # session_id -> {ip, user_agent, created, last_seen}
        self.SESSION_TIMEOUT = 900  # 15 minutes

        # Security events log
        self._events = []
        self._threat_count = 0
        self.MAX_EVENTS = 500

    # ============================================
    # RATE LIMITER
    # ============================================
    def check_rate_limit(self, identifier):
        """Check if request is within rate limits. Returns (allowed, remaining)."""
        now = time.time()
        # Clean old entries
        self._request_log[identifier] = [
            t for t in self._request_log[identifier]
            if now - t < self.RATE_WINDOW
        ]
        count = len(self._request_log[identifier])

        if count >= self.RATE_LIMIT:
            self._log_event("RATE_LIMIT", f"Rate limit exceeded: {identifier}", "HIGH")
            return False, 0

        self._request_log[identifier].append(now)
        return True, self.RATE_LIMIT - count - 1

    # ============================================
    # BRUTE FORCE SHIELD
    # ============================================
    def check_lockout(self, identifier):
        """Check if identifier is locked out. Returns (locked, seconds_remaining)."""
        now = time.time()
        until = self._lockout_until.get(identifier, 0)
        if now < until:
            return True, int(until - now)
        return False, 0

    def record_failed_attempt(self, identifier):
        """Record a failed login attempt and apply progressive lockout."""
        self._failed_attempts[identifier] += 1
        count = self._failed_attempts[identifier]

        # Progressive lockout
        if count >= 3:
            stage = min(count - 3, len(self.LOCKOUT_STAGES) - 1)
            lockout_time = self.LOCKOUT_STAGES[stage]
            self._lockout_until[identifier] = time.time() + lockout_time
            self._log_event(
                "BRUTE_FORCE",
                f"Lockout {lockout_time}s applied to: {identifier} ({count} fails)",
                "CRITICAL"
            )
            self._send_alert(
                f"🚨 *BRUTE FORCE TERDETEKSI*\n\n"
                f"Target: `{identifier}`\n"
                f"Percobaan gagal: {count}x\n"
                f"Lockout: {lockout_time // 60} menit\n"
                f"Waktu: {datetime.now().strftime('%H:%M:%S')}"
            )
            return lockout_time
        return 0

    def record_success(self, identifier):
        """Reset failed attempts on successful login."""
        self._failed_attempts[identifier] = 0
        if identifier in self._lockout_until:
            del self._lockout_until[identifier]

    # ============================================
    # SESSION GUARDIAN
    # ============================================
    def create_session(self, session_id, ip, user_agent):
        """Register a new session."""
        self._active_sessions[session_id] = {
            'ip': ip,
            'user_agent': user_agent,
            'created': time.time(),
            'last_seen': time.time(),
            'fingerprint': self._fingerprint(ip, user_agent),
        }

    def validate_session(self, session_id, ip, user_agent):
        """Validate session integrity. Returns (valid, reason)."""
        session = self._active_sessions.get(session_id)
        if not session:
            return False, "Session not found"

        # Timeout check
        if time.time() - session['last_seen'] > self.SESSION_TIMEOUT:
            del self._active_sessions[session_id]
            return False, "Session expired"

        # Hijack detection — IP or User-Agent changed
        current_fp = self._fingerprint(ip, user_agent)
        if current_fp != session['fingerprint']:
            self._log_event(
                "SESSION_HIJACK",
                f"Session {session_id[:8]}... fingerprint mismatch",
                "CRITICAL"
            )
            self._send_alert(
                f"🚨 *SESSION HIJACK TERDETEKSI*\n\n"
                f"Session: `{session_id[:8]}...`\n"
                f"Original IP: `{session['ip']}`\n"
                f"Current IP: `{ip}`\n"
                f"Waktu: {datetime.now().strftime('%H:%M:%S')}"
            )
            del self._active_sessions[session_id]
            return False, "Session hijack detected"

        # Update last seen
        session['last_seen'] = time.time()
        return True, "OK"

    def _fingerprint(self, ip, user_agent):
        """Generate session fingerprint."""
        raw = f"{ip}|{user_agent}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    # ============================================
    # ANOMALY DETECTOR
    # ============================================
    def detect_anomaly(self, identifier):
        """Detect unusual request patterns."""
        now = time.time()
        timestamps = self._request_log.get(identifier, [])
        if len(timestamps) < 5:
            return False, "Normal"

        # Check for burst — too many requests in very short time
        recent = [t for t in timestamps if now - t < 5]
        if len(recent) > 10:
            self._log_event("ANOMALY", f"Burst detected from {identifier}: {len(recent)} in 5s", "HIGH")
            self._send_alert(
                f"⚠️ *ANOMALY: Burst Request*\n\n"
                f"Source: `{identifier}`\n"
                f"Requests: {len(recent)} dalam 5 detik\n"
                f"Kemungkinan: Bot/Scraper"
            )
            return True, f"Burst: {len(recent)} requests in 5s"

        # Check for uniform intervals (bot pattern)
        if len(timestamps) >= 10:
            intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            recent_intervals = intervals[-10:]
            std = float(__import__('numpy').std(recent_intervals))
            if std < 0.1 and len(recent_intervals) >= 5:
                self._log_event("ANOMALY", f"Bot pattern from {identifier}: uniform intervals", "MEDIUM")
                return True, "Uniform intervals (bot pattern)"

        return False, "Normal"

    # ============================================
    # SECURITY STATUS & LOGS
    # ============================================
    def _log_event(self, event_type, detail, severity="LOW"):
        """Log security event."""
        event = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'type': event_type,
            'detail': detail,
            'severity': severity,
        }
        self._events.append(event)
        if severity in ('HIGH', 'CRITICAL'):
            self._threat_count += 1
        if len(self._events) > self.MAX_EVENTS:
            self._events = self._events[-self.MAX_EVENTS:]

    def get_security_status(self):
        """Get current security status summary."""
        now = time.time()
        active_sessions = sum(
            1 for s in self._active_sessions.values()
            if now - s['last_seen'] < self.SESSION_TIMEOUT
        )
        locked_ips = sum(1 for t in self._lockout_until.values() if t > now)
        recent_events = [e for e in self._events if e.get('severity') in ('HIGH', 'CRITICAL')]

        # Threat level
        if self._threat_count > 10: level = "🔴 CRITICAL"
        elif self._threat_count > 5: level = "🟠 HIGH"
        elif self._threat_count > 0: level = "🟡 MEDIUM"
        else: level = "🟢 AMAN"

        return {
            'threat_level': level,
            'threat_count': self._threat_count,
            'active_sessions': active_sessions,
            'locked_ips': locked_ips,
            'total_events': len(self._events),
            'recent_threats': recent_events[-5:],
        }

    def _send_alert(self, message):
        """Send security alert to Telegram."""
        if not self.telegram_token or not self.chat_id:
            return
        try:
            import requests as req
            req.post(
                f"https://api.telegram.org/bot{self.telegram_token}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'Markdown'
                },
                timeout=5
            )
        except:
            pass

    def get_full_report(self):
        """Full security report for Telegram."""
        s = self.get_security_status()
        lines = [
            f"🛡️ *SECURITY REPORT*\n",
            f"📊 Status: {s['threat_level']}",
            f"🚨 Threats: {s['threat_count']}",
            f"👥 Active Sessions: {s['active_sessions']}",
            f"🔒 Locked IPs: {s['locked_ips']}",
            f"📋 Total Events: {s['total_events']}",
        ]
        if s['recent_threats']:
            lines.append("\n⚠️ *Recent Threats:*")
            for e in s['recent_threats'][-3:]:
                lines.append(f"  • [{e['severity']}] {e['type']}: {e['detail'][:50]}")
        return "\n".join(lines)

    # ============================================
    # CSRF TOKEN PROTECTION
    # ============================================
    def generate_csrf_token(self):
        """Generate a CSRF token for form protection."""
        import secrets
        token = secrets.token_hex(32)
        self._csrf_tokens = getattr(self, '_csrf_tokens', {})
        self._csrf_tokens[token] = time.time()
        # Clean old tokens (>30min)
        self._csrf_tokens = {
            t: ts for t, ts in self._csrf_tokens.items()
            if time.time() - ts < 1800
        }
        return token

    def validate_csrf_token(self, token):
        """Validate CSRF token."""
        tokens = getattr(self, '_csrf_tokens', {})
        if token in tokens:
            if time.time() - tokens[token] < 1800:
                del tokens[token]
                return True
            del tokens[token]
        self._log_event("CSRF", "Invalid or expired CSRF token", "HIGH")
        return False

    # ============================================
    # HONEYPOT TRAP (catches automated bots)
    # ============================================
    def check_honeypot(self, honeypot_value, identifier="unknown"):
        """If honeypot field has content, it's a bot."""
        if honeypot_value:
            self._log_event("HONEYPOT", f"Bot trapped: {identifier}", "CRITICAL")
            self._send_alert(
                f"🍯 *HONEYPOT BOT TERDETEKSI*\n\n"
                f"Source: `{identifier}`\n"
                f"Isi honeypot: `{str(honeypot_value)[:30]}`\n"
                f"Aksi: Banned otomatis"
            )
            return True  # It's a bot
        return False  # Legit user

    # ============================================
    # OTP RATE LIMITER
    # ============================================
    def check_otp_rate(self, identifier, cooldown=60):
        """Prevent OTP spam. Returns (allowed, wait_seconds)."""
        self._otp_log = getattr(self, '_otp_log', defaultdict(list))
        now = time.time()
        # Clean old entries
        self._otp_log[identifier] = [
            t for t in self._otp_log[identifier] if now - t < 300
        ]
        # Max 5 OTP requests per 5 minutes
        if len(self._otp_log[identifier]) >= 5:
            self._log_event("OTP_SPAM", f"OTP spam from {identifier}: {len(self._otp_log[identifier])} in 5min", "HIGH")
            self._send_alert(
                f"⚠️ *OTP SPAM TERDETEKSI*\n\n"
                f"Source: `{identifier}`\n"
                f"OTP requests: {len(self._otp_log[identifier])} dalam 5 menit"
            )
            return False, 300
        # Cooldown between OTP
        if self._otp_log[identifier]:
            last = self._otp_log[identifier][-1]
            if now - last < cooldown:
                return False, int(cooldown - (now - last))
        self._otp_log[identifier].append(now)
        return True, 0

    # ============================================
    # API KEY ROTATION TRACKER
    # ============================================
    def check_key_age(self, key_name, created_date=None):
        """Warn if API key is too old (>90 days)."""
        if not created_date:
            return None
        age_days = (datetime.now() - created_date).days
        if age_days > 90:
            self._log_event("KEY_AGE", f"{key_name} is {age_days} days old", "MEDIUM")
            return f"⚠️ {key_name} sudah {age_days} hari — Pertimbangkan rotate!"
        return None

    # ============================================
    # REQUEST SIGNATURE (anti-tampering)
    # ============================================
    def sign_request(self, data, secret):
        """Sign request data for integrity check."""
        payload = json.dumps(data, sort_keys=True)
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def verify_request(self, data, signature, secret):
        """Verify request signature."""
        expected = self.sign_request(data, secret)
        if not hmac.compare_digest(expected, signature):
            self._log_event("TAMPER", "Request signature mismatch", "CRITICAL")
            return False
        return True

