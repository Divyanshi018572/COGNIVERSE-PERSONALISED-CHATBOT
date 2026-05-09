# auth/db_queries.py
# Auth-specific database queries.
# Uses the EXISTING pool from core/db.py — does not create a new connection.

from core.db import pool          # ← uses existing psycopg_pool
from auth.security import hash_password, verify_password
from auth.models import RegisterRequest
from typing import Optional


def get_user_by_username(username: str) -> Optional[dict]:
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE username = %s AND is_active = TRUE",
                (username.lower(),)
            )
            row = cursor.fetchone()
            if row:
                cols = [desc[0] for desc in cursor.description]
                return dict(zip(cols, row))
            return None


def get_user_by_email(email: str) -> Optional[dict]:
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE email = %s",
                (email.lower(),)
            )
            row = cursor.fetchone()
            if row:
                cols = [desc[0] for desc in cursor.description]
                return dict(zip(cols, row))
            return None


def create_user(req: RegisterRequest) -> dict:
    """
    Insert a new user. Raises ValueError if username/email already exists.
    Returns the created user row as dict.
    """
    # Check uniqueness before inserting
    if get_user_by_username(req.username):
        raise ValueError("Username already taken")
    if get_user_by_email(req.email):
        raise ValueError("Email already registered")

    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO users (username, email, full_name, hashed_password)
                VALUES (%s, %s, %s, %s)
                RETURNING id, username, email, full_name, created_at
                """,
                (req.username.lower(), req.email.lower(), req.full_name, hash_password(req.password))
            )
            row = cursor.fetchone()
            cols = [desc[0] for desc in cursor.description]
            return dict(zip(cols, row))


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """
    Verify username + password.
    Returns user dict on success, None on failure.
    """
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user
