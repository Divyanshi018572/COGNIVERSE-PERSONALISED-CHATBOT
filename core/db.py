import os
from psycopg_pool import ConnectionPool
from contextlib import contextmanager

POSTGRES_URI = os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/chatbot_db")

pool = ConnectionPool(
    conninfo=POSTGRES_URI,
    max_size=20,
    kwargs={"autocommit": True}
)

def init_db():
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            # Drop old tables to avoid schema conflicts during development
            cursor.execute("DROP TABLE IF EXISTS user_memory CASCADE")
            cursor.execute("DROP TABLE IF EXISTS users CASCADE")
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id          SERIAL PRIMARY KEY,
                    username    TEXT UNIQUE NOT NULL,
                    email       TEXT UNIQUE NOT NULL,
                    full_name   TEXT NOT NULL,
                    hashed_password TEXT NOT NULL,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    is_active   BOOLEAN DEFAULT TRUE
                )
            """)
            # Memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_memory (
                    id SERIAL PRIMARY KEY,
                    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                    fact TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Ensure default user 1 exists for open-access mode
            cursor.execute("""
                INSERT INTO users (id, username, email, full_name, hashed_password, is_active) 
                VALUES (1, 'default', 'default@cognibot.local', 'Default User', 'default', TRUE) 
                ON CONFLICT (id) DO NOTHING
            """)

# --- Auth Helpers ---
def get_user_by_email(email: str):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id, email, password_hash, is_verified FROM users WHERE email = %s", (email,))
            return cursor.fetchone()

def create_user(email: str, password_hash: str, provider: str = 'local', is_verified: bool = False, google_id: str = None):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO users (email, password_hash, auth_provider, is_verified, google_id) VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (email, password_hash, provider, is_verified, google_id)
            )
            return cursor.fetchone()[0]

def set_verification_code(user_id: int, code: str, expires_at):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET verification_code = %s, verification_expires_at = %s WHERE id = %s",
                (code, expires_at, user_id)
            )

def verify_user(user_id: int):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET is_verified = TRUE, verification_code = NULL, verification_expires_at = NULL WHERE id = %s",
                (user_id,)
            )

def get_verification_details(user_id: int):
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT verification_code, verification_expires_at FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()

# --- Memory Helpers ---
def save_facts(user_id: int, facts_list: list[str]):
    if not facts_list:
        return
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            for fact in facts_list:
                cursor.execute(
                    "INSERT INTO user_memory (user_id, fact) VALUES (%s, %s)",
                    (user_id, fact)
                )

def get_facts(user_id: int) -> str:
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT fact FROM user_memory WHERE user_id = %s", (user_id,))
            rows = cursor.fetchall()
            if not rows:
                return ""
            return "\n".join([f"- {row[0]}" for row in rows])
