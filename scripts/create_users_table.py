import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def create_users_table():
    db_url = os.getenv("POSTGRES_URI", "postgresql://user:password@localhost:5432/chatbot_db")
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS users CASCADE;")
            cur.execute("""
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
        conn.commit()
    print("✓ users table ready")

if __name__ == "__main__":
    create_users_table()
