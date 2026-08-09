"""Passphrase workspaces. No accounts: the phrase locates AND unlocks.

  workspace_id = base32( HKDF-SHA256(phrase, salt=pepper, info='ws-id') )
      deterministic -> the same phrase always finds the same workspace,
      nothing enumerable, nothing stored to look it up;
  verifier     = scrypt(phrase, random salt || pepper)   [stdlib, n=2^14]
      stored -> the id alone must never authenticate.

Bearer tokens (HMAC over id|expiry, keyed from the pepper) let hot paths
skip the KDF; either `Authorization: Bearer <t>` or
`Authorization: Passphrase <b64 phrase>` is accepted everywhere.
"""
from __future__ import annotations
import base64, hashlib, hmac, os, re, time, unicodedata
from dataclasses import dataclass
from . import config

class AuthError(Exception):
    def __init__(self, status: int, detail: str):
        self.status, self.detail = status, detail

def normalize(phrase: str) -> str:
    p = unicodedata.normalize('NFKC', phrase or '').strip()
    return re.sub(r'\s+', ' ', p)

def validate_strength(phrase: str):
    if len(phrase) >= 20: return
    if len(phrase.split(' ')) >= 4: return
    raise AuthError(400, 'passphrase too weak: use at least 4 words or 20 characters')

def workspace_id(phrase: str) -> str:
    prk = hmac.new(config.PEPPER.encode(), phrase.encode(), hashlib.sha256).digest()
    okm = hmac.new(prk, b'relspec-ws-id\x01', hashlib.sha256).digest()
    return base64.b32encode(okm[:10]).decode().lower().rstrip('=')

def make_verifier(phrase: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(phrase.encode(), salt=salt+config.PEPPER.encode(),
                       n=2**14, r=8, p=1, dklen=32)
    return f'scrypt$16384$8$1${salt.hex()}${h.hex()}'

def check_verifier(phrase: str, verifier: str) -> bool:
    try:
        _, n, r, p, salt_hex, h_hex = verifier.split('$')
        h = hashlib.scrypt(phrase.encode(),
                           salt=bytes.fromhex(salt_hex)+config.PEPPER.encode(),
                           n=int(n), r=int(r), p=int(p), dklen=32)
        return hmac.compare_digest(h.hex(), h_hex)
    except Exception:
        return False

def _token_key() -> bytes:
    return hashlib.sha256(b'relspec-token|'+config.PEPPER.encode()).digest()

def mint_token(wsid: str, now: float | None = None) -> str:
    exp = int((now or time.time())+config.TOKEN_TTL_S)
    msg = f'{wsid}|{exp}'.encode()
    sig = hmac.new(_token_key(), msg, hashlib.sha256).digest()[:20]
    return base64.urlsafe_b64encode(msg+b'|'+sig).decode()

def check_token(token: str) -> str | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        msg, sig = raw.rsplit(b'|', 1)
        if not hmac.compare_digest(
                hmac.new(_token_key(), msg, hashlib.sha256).digest()[:20], sig):
            return None
        wsid, exp = msg.decode().split('|')
        if time.time() > int(exp): return None
        return wsid
    except Exception:
        return None

# Derived API keys: same bearer shape, separate HMAC context, no expiry —
# revocation lives in the api_key table, which authenticate() consults.
def _key_token_key() -> bytes:
    return hashlib.sha256(b'relspec-key|'+config.PEPPER.encode()).digest()

def mint_key_token(wsid: str, key_id: str, scope: str) -> str:
    msg = f'{wsid}|{key_id}|{scope}'.encode()
    sig = hmac.new(_key_token_key(), msg, hashlib.sha256).digest()[:20]
    return base64.urlsafe_b64encode(msg+b'|'+sig).decode()

def check_key_token(token: str) -> tuple[str, str, str] | None:
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        msg, sig = raw.rsplit(b'|', 1)
        if not hmac.compare_digest(
                hmac.new(_key_token_key(), msg, hashlib.sha256).digest()[:20], sig):
            return None
        wsid, key_id, scope = msg.decode().split('|')
        if scope not in SCOPES: return None
        return wsid, key_id, scope
    except Exception:
        return None

# in-memory failure throttle: 5 bad passphrase attempts / minute / client
_fails: dict[str, list[float]] = {}
def throttle(client: str):
    now = time.time()
    lst = [t for t in _fails.get(client, []) if now-t < 60]
    _fails[client] = lst
    if len(lst) >= 5:
        raise AuthError(429, 'too many failed attempts; wait a minute')
def record_fail(client: str):
    _fails.setdefault(client, []).append(time.time())

SCOPES = ('read', 'ingest', 'full')

@dataclass
class Principal:
    workspace_id: str
    scope: str = 'full'
    key_id: str | None = None

def require_scope(pr: Principal, need: str):
    """'full' does everything; 'read'/'ingest' keys only their own lane."""
    if pr.scope != 'full' and pr.scope != need:
        raise AuthError(403, f'{pr.scope}-scoped key cannot perform this operation')

def authenticate(header: str | None, cur, client: str) -> Principal:
    """cur: open cursor for verifier / key lookup. Raises AuthError."""
    if not header:
        raise AuthError(401, 'missing Authorization header')
    kind, _, value = header.partition(' ')
    kind = kind.lower(); value = value.strip()
    if kind == 'bearer':
        wsid = check_token(value)
        if wsid: return Principal(wsid)
        kt = check_key_token(value)
        if kt:
            wsid, key_id, scope = kt
            row = cur.execute(
                'SELECT revoked FROM api_key WHERE key_id=%s AND workspace_id=%s',
                (key_id, wsid)).fetchone()
            if not row or row[0]:
                raise AuthError(401, 'key revoked or unknown')
            return Principal(wsid, scope, key_id)
        raise AuthError(401, 'invalid or expired token')
    if kind == 'passphrase':
        throttle(client)
        try:
            phrase = normalize(base64.b64decode(value).decode())
        except Exception:
            raise AuthError(400, 'passphrase header must be base64')
        wsid = workspace_id(phrase)
        row = cur.execute('SELECT auth_hash FROM workspace WHERE workspace_id=%s',
                          (wsid,)).fetchone()
        if not row or not check_verifier(phrase, row[0]):
            record_fail(client)
            raise AuthError(401, 'unknown workspace or wrong passphrase')
        return Principal(wsid)
    raise AuthError(401, 'use Bearer <token> or Passphrase <base64 phrase>')
