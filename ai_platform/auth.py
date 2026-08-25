"""Local authentication + RBAC (continuous-engineering priority #24).

Real, not a stub: SQLite user table, PBKDF2 password hashing (stdlib
hashlib, no new dependency), HMAC-signed session tokens (stdlib hmac +
secrets). Distinct from third-party SSO (Google/Okta/etc.), which is
genuinely BLOCKED - that needs registering an OAuth application with an
external provider and credentials this environment doesn't have. Local
username/password + role-based permission checks needs neither.
"""

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time

DB_PATH = os.path.join("data", "auth.db")
SESSION_SECRET_PATH = os.path.join("data", ".session_secret")
SESSION_TTL_SECONDS = 8 * 3600

ROLES = ("admin", "analyst", "viewer")

# What each role may do. Checked by require_permission() before any
# capability the orchestrator considers "high-impact" runs.
ROLE_PERMISSIONS = {
    "admin": {"read", "write", "approve", "manage_users"},
    "analyst": {"read", "write"},
    "viewer": {"read"},
}


class AuthError(Exception):
    pass


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        "username TEXT PRIMARY KEY, password_hash TEXT, salt TEXT, "
        "role TEXT, created_at TEXT)"
    )
    return conn


def _get_session_secret() -> bytes:
    if os.path.exists(SESSION_SECRET_PATH):
        with open(SESSION_SECRET_PATH, "rb") as f:
            return f.read()
    os.makedirs(os.path.dirname(SESSION_SECRET_PATH), exist_ok=True)
    secret = secrets.token_bytes(32)
    with open(SESSION_SECRET_PATH, "wb") as f:
        f.write(secret)
    return secret


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def create_user(username: str, password: str, role: str = "viewer"):
    if role not in ROLES:
        raise AuthError(f"Unknown role '{role}'. Use one of {ROLES}")
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters")

    conn = _get_connection()
    try:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            raise AuthError(f"User '{username}' already exists")
        salt = secrets.token_hex(16)
        pw_hash = _hash_password(password, salt)
        import datetime
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, created_at) VALUES (?,?,?,?,?)",
            (username, pw_hash, salt, role, datetime.datetime.utcnow().isoformat() + "Z"),
        )
        conn.commit()
        return {"username": username, "role": role}
    finally:
        conn.close()


def delete_user(username: str):
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
    finally:
        conn.close()


def authenticate(username: str, password: str) -> dict:
    conn = _get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if row is None:
            raise AuthError("Invalid username or password")
        expected = _hash_password(password, row["salt"])
        # Constant-time comparison - a naive `==` here leaks timing
        # information about how many leading characters matched.
        if not hmac.compare_digest(expected, row["password_hash"]):
            raise AuthError("Invalid username or password")
        return {"username": row["username"], "role": row["role"]}
    finally:
        conn.close()


def issue_session_token(username: str, role: str) -> str:
    """HMAC-signed token: `username.role.expiry.signature`. Not a JWT
    library (avoids the dependency) but the same real idea: the payload
    can't be forged without the server-side secret, and it self-expires."""
    expiry = int(time.time()) + SESSION_TTL_SECONDS
    payload = f"{username}.{role}.{expiry}"
    signature = hmac.new(_get_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{signature}"


def verify_session_token(token: str) -> dict:
    try:
        username, role, expiry_str, signature = token.split(".")
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        raise AuthError("Malformed session token")

    payload = f"{username}.{role}.{expiry}"
    expected_sig = hmac.new(_get_session_secret(), payload.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected_sig, signature):
        raise AuthError("Invalid session token signature")
    if time.time() > expiry:
        raise AuthError("Session token expired")
    if role not in ROLES:
        raise AuthError("Unknown role in token")
    return {"username": username, "role": role}


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())


def require_permission(token: str, permission: str) -> dict:
    """Raises AuthError if the token is invalid/expired or the role lacks
    the permission. Returns the session dict on success."""
    session = verify_session_token(token)
    if not has_permission(session["role"], permission):
        raise AuthError(f"Role '{session['role']}' lacks permission '{permission}'")
    return session
