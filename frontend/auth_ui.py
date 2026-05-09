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
