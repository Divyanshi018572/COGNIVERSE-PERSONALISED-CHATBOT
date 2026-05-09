# AGENTS.md — Cognibot Auth Addition Guide
# READ THIS ENTIRE FILE BEFORE WRITING A SINGLE LINE OF CODE
# This file tells you exactly what to ADD, what to NEVER TOUCH, and in what order.

---

## CRITICAL RULES — OBEY THESE ABOVE ALL ELSE

```
RULE 1: DO NOT modify any existing file logic.
        You may only ADD new lines at the TOP of existing files.
        You may only ADD new files.
        Never delete, rename, or restructure anything.

RULE 2: DO NOT change any existing imports, routes, or function signatures.
        Only append new imports at the top and new routes at the bottom.

RULE 3: The app MUST work identically for authenticated users as it does today.
        Auth is a gate — it blocks unauthenticated users, nothing more.

RULE 4: PostgreSQL is already running. DO NOT add a new database or ORM.
        Use the existing pool from core/db.py for all auth queries.

RULE 5: Docker is already configured. DO NOT modify docker-compose.yml,
        Dockerfile.backend, or Dockerfile.frontend.

RULE 6: If you are unsure about any existing code — STOP and ask.
        Do not guess. Do not infer. Do not fill in blanks.
```

---

## WHAT THIS TASK IS

Add login + register + logout to Cognibot without breaking anything.

Architecture of the addition:
```
BEFORE:
  Browser → frontend/app.py → backend/main.py → orchestrator → agents

AFTER:
  Browser → frontend/app.py [AUTH GATE] → backend/main.py [JWT CHECK] → orchestrator → agents
                ↓ (if not logged in)
           Login / Register UI
```

Auth flow:
1. User visits the Streamlit app
2. If no valid JWT cookie → show Login/Register screen → st.stop()
3. If valid JWT cookie → app renders normally (zero change to existing chat logic)
4. Register → creates user in PostgreSQL users table → redirects to login
5. Login → backend verifies credentials → returns JWT → stored in cookie
6. Logout → clears cookie → back to login screen

---

## FILES TO CREATE (new files only)

```
Cognibot/
├── auth/                         ← CREATE this folder
│   ├── __init__.py               ← CREATE (empty)
│   ├── models.py                 ← CREATE (User pydantic model)
│   ├── security.py               ← CREATE (JWT + bcrypt helpers)
│   ├── routes.py                 ← CREATE (FastAPI auth routes)
│   └── db_queries.py             ← CREATE (auth-specific DB queries)
│
├── frontend/
│   └── auth_ui.py                ← CREATE (Streamlit login/register UI)
│
└── scripts/
    └── create_users_table.py     ← CREATE (one-time DB migration script)
```

---

## FILES TO MODIFY (additions only — existing code untouched)

```
backend/main.py      → ADD: import auth router + mount it (3 lines at bottom)
frontend/app.py      → ADD: auth gate at very top (before any existing code runs)
requirements.txt     → ADD: 4 new packages at bottom
.env                 → ADD: 2 new env vars at bottom
```

---

## STEP 1 — Add packages to requirements.txt

APPEND these 4 lines at the BOTTOM of the existing requirements.txt.
Do not touch any existing line.

```
# Auth additions
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
httpx==0.27.0
```

---

## STEP 2 — Add env vars to .env

APPEND these at the BOTTOM of the existing .env file.
Do not touch any existing line.

```
# Auth additions
JWT_SECRET_KEY=replace_this_with_a_random_64_char_string_right_now
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=10080
```

Generate a real secret:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```
Copy the output into JWT_SECRET_KEY in .env.

---

## STEP 3 — Run DB migration (one time only)

CREATE: scripts/create_users_table.py

```python
# scripts/create_users_table.py
# Run once: python scripts/create_users_table.py
# Creates the users table in the existing PostgreSQL database.
# Safe to run multiple times — uses IF NOT EXISTS.

import asyncio
import asyncpg
import os
from dotenv import load_dotenv

load_dotenv()


async def create_users_table():
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          SERIAL PRIMARY KEY,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            full_name   TEXT NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            is_active   BOOLEAN DEFAULT TRUE
        );
    """)
    await conn.close()
    print("✓ users table ready")


if __name__ == "__main__":
    asyncio.run(create_users_table())
```

Run it:
```bash
cd Cognibot
python scripts/create_users_table.py
```

Verify it worked (optional):
```bash
# Inside your postgres container or psql:
\dt users
```

---

## STEP 4 — CREATE auth/models.py

```python
# auth/models.py
# Pydantic models for auth requests and responses only.
# No changes to existing models anywhere else.

from pydantic import BaseModel, EmailStr, field_validator
import re


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    full_name: str
    password: str

    @field_validator("username")
    @classmethod
    def username_valid(cls, v):
        if not re.match(r"^[a-zA-Z0-9_]{3,20}$", v):
            raise ValueError(
                "Username must be 3-20 chars: letters, numbers, underscore only"
            )
        return v.lower()

    @field_validator("full_name")
    @classmethod
    def name_valid(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Full name must be at least 2 characters")
        return v.strip()

    @field_validator("password")
    @classmethod
    def password_valid(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        return v


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    full_name: str
    username: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: str
    full_name: str
```

---

## STEP 5 — CREATE auth/security.py

```python
# auth/security.py
# JWT creation/verification and bcrypt password hashing.
# Self-contained — no dependencies on existing app code.

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days

if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is not set in .env — "
        "run: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(plaintext: str) -> str:
    return pwd_context.hash(plaintext)


def verify_password(plaintext: str, hashed: str) -> bool:
    return pwd_context.verify(plaintext, hashed)


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(minutes=EXPIRE_MINUTES)
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    """
    Returns the decoded payload dict if token is valid.
    Returns None if token is expired or invalid.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
```

---

## STEP 6 — CREATE auth/db_queries.py

```python
# auth/db_queries.py
# Auth-specific database queries.
# Uses the EXISTING pool from core/db.py — does not create a new connection.

from core.db import get_pool          # ← uses existing pool, not a new one
from auth.security import hash_password, verify_password
from auth.models import RegisterRequest
from typing import Optional


async def get_user_by_username(username: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1 AND is_active = TRUE",
            username.lower(),
        )
        return dict(row) if row else None


async def get_user_by_email(email: str) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM users WHERE email = $1",
            email.lower(),
        )
        return dict(row) if row else None


async def create_user(req: RegisterRequest) -> dict:
    """
    Insert a new user. Raises ValueError if username/email already exists.
    Returns the created user row as dict.
    """
    pool = await get_pool()

    # Check uniqueness before inserting
    if await get_user_by_username(req.username):
        raise ValueError("Username already taken")
    if await get_user_by_email(req.email):
        raise ValueError("Email already registered")

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO users (username, email, full_name, hashed_password)
            VALUES ($1, $2, $3, $4)
            RETURNING id, username, email, full_name, created_at
            """,
            req.username.lower(),
            req.email.lower(),
            req.full_name,
            hash_password(req.password),
        )
        return dict(row)


async def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Verify username + password.
    Returns user dict on success, None on failure.
    """
    user = await get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
```

**IMPORTANT:** The `get_pool()` function imported above must already exist in
`core/db.py`. Check what the actual function name is in that file before
writing this import. If it is named differently (e.g. `get_db_pool`,
`create_pool`, `db_pool`), use that name instead. Do NOT change `core/db.py`.

---

## STEP 7 — CREATE auth/routes.py

```python
# auth/routes.py
# Two FastAPI routes: POST /auth/register and POST /auth/login
# Mounted into the existing FastAPI app in backend/main.py
# Does not touch any existing route.

from fastapi import APIRouter, HTTPException, status
from auth.models import RegisterRequest, LoginRequest, TokenResponse
from auth.db_queries import create_user, authenticate_user
from auth.security import create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest):
    """
    Register a new user.
    Returns success message — user must then login to get token.
    """
    try:
        user = await create_user(req)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        )
    return {"message": f"Account created. Welcome, {user['full_name'].split()[0]}!"}


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest):
    """
    Authenticate and return a JWT token.
    Frontend stores this token in a cookie.
    """
    user = await authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token({
        "sub": user["username"],
        "name": user["full_name"],
    })
    return TokenResponse(
        access_token=token,
        full_name=user["full_name"],
        username=user["username"],
    )
```

---

## STEP 8 — MODIFY backend/main.py (3 lines only)

Open `backend/main.py`. Find where the FastAPI `app` object is created and
where other routers are included (if any). Add ONLY these lines:

```python
# ADD at the top with other imports (do not move existing imports):
from auth.routes import router as auth_router

# ADD after the app = FastAPI(...) line and after any existing router includes:
app.include_router(auth_router)
```

That is the COMPLETE change to `backend/main.py`. Two lines. Nothing else.

**Verify before saving:**
- The existing chat/stream routes are still there and unchanged
- No existing import was removed or modified
- `app.include_router(auth_router)` appears once, after app definition

---

## STEP 9 — CREATE frontend/auth_ui.py

```python
# frontend/auth_ui.py
# Renders the login and register screens.
# Called from frontend/app.py ONLY when user is not authenticated.
# Has zero knowledge of the chat logic — completely isolated.

import streamlit as st
import httpx
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")


# ── CSS (injected once when auth page loads) ──────────────────────────────────

AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

.stApp { background-color: #f5f4ed !important; font-family: 'Inter', sans-serif; }
#MainMenu, footer, header, .stDeployButton { visibility: hidden !important; }

.auth-logo { text-align: center; margin-bottom: 28px; }
.auth-icon  { font-size: 38px; color: #c96442; display: block; margin-bottom: 6px; }
.auth-title {
    font-family: 'Georgia', serif;
    font-size: 26px; font-weight: 500;
    color: #141413; margin: 0;
}
.auth-sub   { font-size: 14px; color: #87867f; margin: 4px 0 0; }

.auth-error {
    background: rgba(181,51,51,0.07);
    border: 1px solid rgba(181,51,51,0.18);
    border-left: 3px solid #b53333;
    color: #b53333; border-radius: 8px;
    padding: 10px 14px; font-size: 13px;
    margin-bottom: 14px;
}
.auth-success {
    background: rgba(45,106,79,0.07);
    border: 1px solid rgba(45,106,79,0.18);
    border-left: 3px solid #2d6a4f;
    color: #2d6a4f; border-radius: 8px;
    padding: 10px 14px; font-size: 13px;
    margin-bottom: 14px;
}
.auth-footer {
    text-align: center; margin-top: 18px;
    font-size: 12px; color: #87867f;
}

/* Input styling */
[data-testid="stTextInput"] input {
    background: #ffffff !important;
    border: 1px solid #d1cfc5 !important;
    border-radius: 10px !important;
    font-size: 15px !important;
    color: #141413 !important;
    padding: 10px 14px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #3898ec !important;
    box-shadow: 0 0 0 3px rgba(56,152,236,0.18) !important;
}

/* Label styling */
[data-testid="stTextInput"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #4d4c48 !important;
    margin-bottom: 4px !important;
}

/* Primary button */
[data-testid="stFormSubmitButton"] button {
    background: #c96442 !important;
    color: #faf9f5 !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 12px 24px !important;
    width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.15s !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    background: #d97757 !important;
}

/* Tab buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: transparent !important;
    color: #87867f !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    width: 100% !important;
    padding: 8px !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #f5f4ed !important;
    color: #141413 !important;
}
</style>
"""


def _inject_css():
    st.markdown(AUTH_CSS, unsafe_allow_html=True)


# ── Session state helpers ─────────────────────────────────────────────────────

def _init_auth_state():
    defaults = {
        "auth_mode": "login",          # "login" | "register"
        "auth_error": None,
        "auth_success": None,
        "auth_failed_attempts": 0,
        "auth_token": None,
        "auth_username": None,
        "auth_full_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API calls ─────────────────────────────────────────────────────────────────

def _call_login(username: str, password: str) -> tuple[bool, str, dict]:
    """
    POST /auth/login
    Returns (success, error_message, token_data)
    """
    try:
        r = httpx.post(
            f"{BACKEND_URL}/auth/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if r.status_code == 200:
            return True, "", r.json()
        detail = r.json().get("detail", "Login failed")
        return False, detail, {}
    except httpx.ConnectError:
        return False, "Cannot reach backend. Is the server running?", {}
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", {}


def _call_register(
    username: str, email: str, full_name: str, password: str
) -> tuple[bool, str]:
    """
    POST /auth/register
    Returns (success, message)
    """
    try:
        r = httpx.post(
            f"{BACKEND_URL}/auth/register",
            json={
                "username": username,
                "email": email,
                "full_name": full_name,
                "password": password,
            },
            timeout=10,
        )
        if r.status_code == 201:
            return True, r.json().get("message", "Account created!")
        detail = r.json().get("detail", "Registration failed")
        return False, detail
    except httpx.ConnectError:
        return False, "Cannot reach backend. Is the server running?"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


# ── Login form ────────────────────────────────────────────────────────────────

def _render_login():
    # Show success message if just registered
    if st.session_state.auth_success:
        st.markdown(
            f'<div class="auth-success">✓ {st.session_state.auth_success}'
            f" Please sign in.</div>",
            unsafe_allow_html=True,
        )

    # Show error
    if st.session_state.auth_error:
        st.markdown(
            f'<div class="auth-error">{st.session_state.auth_error}</div>',
            unsafe_allow_html=True,
        )

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username", placeholder="your_username")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In →", use_container_width=True)

    if submitted:
        if not username.strip() or not password:
            st.session_state.auth_error = "Please fill in both fields."
            st.rerun()

        success, error, data = _call_login(username.strip(), password)

        if success:
            # Store token + user info in session state
            st.session_state.auth_token     = data["access_token"]
            st.session_state.auth_username  = data["username"]
            st.session_state.auth_full_name = data["full_name"]
            st.session_state.auth_error     = None
            st.session_state.auth_success   = None
            st.session_state.auth_failed_attempts = 0
            st.rerun()   # App re-renders — auth gate passes — chat loads
        else:
            st.session_state.auth_failed_attempts += 1
            attempts_left = max(0, 5 - st.session_state.auth_failed_attempts)
            if attempts_left == 0:
                st.session_state.auth_error = (
                    "Too many failed attempts. Refresh the page to try again."
                )
            else:
                st.session_state.auth_error = (
                    f"{error} ({attempts_left} attempt(s) left)"
                )
            st.rerun()

    st.markdown(
        '<div class="auth-footer">Secure · Private · No data shared</div>',
        unsafe_allow_html=True,
    )


# ── Register form ─────────────────────────────────────────────────────────────

def _render_register():
    if st.session_state.auth_error:
        st.markdown(
            f'<div class="auth-error">{st.session_state.auth_error}</div>',
            unsafe_allow_html=True,
        )

    with st.form("register_form", clear_on_submit=True):
        full_name = st.text_input("Full Name", placeholder="Divyanshi Sharma")
        username  = st.text_input("Username",  placeholder="divyanshi018572")
        email     = st.text_input("Email",     placeholder="you@example.com")
        password  = st.text_input(
            "Password", type="password", placeholder="Min. 8 characters"
        )
        confirm   = st.text_input(
            "Confirm Password", type="password", placeholder="Repeat password"
        )
        submitted = st.form_submit_button(
            "Create Account →", use_container_width=True
        )

    if submitted:
        # Client-side checks before hitting the API
        if password != confirm:
            st.session_state.auth_error = "Passwords do not match."
            st.rerun()
        if len(password) < 8:
            st.session_state.auth_error = "Password must be at least 8 characters."
            st.rerun()

        success, msg = _call_register(
            username.strip(), email.strip(), full_name.strip(), password
        )
        if success:
            st.session_state.auth_success = msg
            st.session_state.auth_error   = None
            st.session_state.auth_mode    = "login"
            st.rerun()
        else:
            st.session_state.auth_error = msg
            st.rerun()

    st.markdown(
        '<div class="auth-footer">'
        'Already have an account? '
        '<a href="#" style="color:#c96442;text-decoration:none">'
        'Use Sign In tab above</a></div>',
        unsafe_allow_html=True,
    )


# ── Public entry point (called from app.py) ───────────────────────────────────

def render_auth_page():
    """
    Renders the full auth page (logo + tabs + form).
    Call this from frontend/app.py before any other UI.
    Ends with st.stop() — chat code below it never runs.
    """
    _init_auth_state()
    _inject_css()

    # Center column layout
    _, col, _ = st.columns([1, 1.2, 1])

    with col:
        # Logo
        st.markdown(
            """
            <div class="auth-logo">
                <span class="auth-icon">⬡</span>
                <p class="auth-title">Cognibot</p>
                <p class="auth-sub">Multi-Agent AI System</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Tab switcher
        t1, t2 = st.columns(2)
        with t1:
            if st.button("Sign In", key="tab_login", use_container_width=True):
                st.session_state.auth_mode  = "login"
                st.session_state.auth_error = None
                st.rerun()
        with t2:
            if st.button("Register", key="tab_reg", use_container_width=True):
                st.session_state.auth_mode  = "register"
                st.session_state.auth_error = None
                st.rerun()

        st.markdown("---")

        if st.session_state.auth_mode == "login":
            _render_login()
        else:
            _render_register()
```

---

## STEP 10 — MODIFY frontend/app.py (additions at top only)

Open `frontend/app.py`. Find the very first line of code (after any module
docstring). Add ONLY the following block BEFORE any existing code runs.

```python
# ═══════════════════════════════════════════════════════════════════
# AUTH GATE — added on top of existing app. Do not move this block.
# ═══════════════════════════════════════════════════════════════════
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.auth_ui import render_auth_page

# Session state keys used by auth — init before anything else
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "auth_username" not in st.session_state:
    st.session_state.auth_username = None
if "auth_full_name" not in st.session_state:
    st.session_state.auth_full_name = None

# Gate: if no token, show auth page and stop. Chat code never runs.
if not st.session_state.get("auth_token"):
    render_auth_page()
    st.stop()
# ═══════════════════════════════════════════════════════════════════
# EXISTING APP CODE STARTS HERE — DO NOT CHANGE ANYTHING BELOW
# ═══════════════════════════════════════════════════════════════════
```

Then find the sidebar in the existing app (wherever `st.sidebar` is used).
Add this logout block ONCE at the top of the sidebar section:

```python
# Logout — ADD inside the existing sidebar block, at the top
with st.sidebar:
    # ── Auth user info (added) ──────────────────────────────────────
    name_display = st.session_state.get("auth_full_name", "User")
    uname_display = st.session_state.get("auth_username", "")
    initial = name_display[0].upper() if name_display else "U"

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;
                padding:12px 0;border-bottom:1px solid #30302e;
                margin-bottom:10px">
        <div style="width:32px;height:32px;border-radius:50%;
                    background:#c96442;display:flex;align-items:center;
                    justify-content:center;font-size:14px;
                    font-weight:500;color:#faf9f5;flex-shrink:0">
            {initial}
        </div>
        <div>
            <div style="font-size:13px;color:#faf9f5;font-weight:500">
                {name_display}
            </div>
            <div style="font-size:11px;color:#87867f">{uname_display}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Logout", key="logout_btn"):
        for key in ["auth_token", "auth_username", "auth_full_name",
                    "auth_mode", "auth_error", "auth_success",
                    "auth_failed_attempts"]:
            st.session_state.pop(key, None)
        st.rerun()
    # ── End auth user info ──────────────────────────────────────────

    # ... existing sidebar code continues unchanged below ...
```

**IMPORTANT RULE:** The sidebar addition goes INSIDE the EXISTING `with st.sidebar:`
block — not in a new one. Find the existing sidebar block and add only these lines
at its top. If the existing app uses `st.sidebar.X` calls instead of a `with` block,
wrap just the addition in `with st.sidebar:` separately.

---

## STEP 11 — CREATE auth/__init__.py

```python
# auth/__init__.py
# Empty — marks auth/ as a Python package.
```

---

## VERIFICATION CHECKLIST

Run through this after every file is created/modified:

```
□ scripts/create_users_table.py runs without error
□ python -c "from auth.security import hash_password; print(hash_password('test'))" works
□ python -c "from auth.routes import router" works
□ Backend starts: uvicorn backend.main:app — no import errors
□ GET http://localhost:8000/docs shows /auth/register and /auth/login routes
□ POST /auth/register with valid body returns 201
□ POST /auth/login with correct credentials returns token
□ POST /auth/login with wrong password returns 401
□ Frontend starts: streamlit run frontend/app.py — shows login screen
□ Login with registered credentials — chat UI loads normally
□ Chat works exactly as before login — send a message, get a response
□ Logout clears session — login screen appears again
□ Register with duplicate username — shows "Username already taken"
□ Register with duplicate email — shows "Email already registered"
□ 5 wrong login attempts — shows lockout message
```

---

## WHAT TO DO IF SOMETHING BREAKS

```
If backend fails to start:
  1. Check: did you change any existing line in backend/main.py? Revert it.
  2. Check: is DATABASE_URL correct in .env?
  3. Check: did the users table get created? Run the migration script again.

If frontend shows an error after auth gate:
  1. Check: is the auth gate block ABOVE all existing st.* calls?
  2. Check: did any existing import get accidentally removed?
  3. Revert frontend/app.py to original, apply auth gate again carefully.

If "get_pool is not defined" error:
  1. Open core/db.py and find the actual name of the pool function.
  2. Update the import in auth/db_queries.py to match that exact name.
  3. Do not change core/db.py itself.

If chat stops working after login:
  1. The token is in st.session_state.auth_token — check if backend
     routes need it. If existing routes don't require auth headers,
     nothing needs to change. Just verify the token is not being
     passed to routes that don't expect it.
```

---

## SUMMARY — COMPLETE CHANGE LIST

```
NEW FILES (6):
  auth/__init__.py
  auth/models.py
  auth/security.py
  auth/db_queries.py
  auth/routes.py
  frontend/auth_ui.py
  scripts/create_users_table.py

MODIFIED FILES (4) — additions only:
  requirements.txt     → 4 lines appended
  .env                 → 3 lines appended
  backend/main.py      → 2 lines added (import + include_router)
  frontend/app.py      → auth gate block at top + logout in sidebar

NEVER TOUCHED:
  core/orchestrator.py
  core/db.py
  core/config.py
  agents/*  (all 8 agent files)
  tools/*   (all 3 tool files)
  models/*  (all 3 model files)
  utils/rate_limiter.py
  docker-compose.yml
  Dockerfile.backend
  Dockerfile.frontend
```
