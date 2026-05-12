import json
import sqlite3
from datetime import datetime
from utils.logger import get_logger
import os

logger = get_logger(__name__)

_conn: sqlite3.Connection | None = None
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "feedback.db")

def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id   TEXT NOT NULL,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                thumbs_up   INTEGER DEFAULT 0,
                thumbs_down INTEGER DEFAULT 0,
                user_comment TEXT,
                eval_score  REAL,
                timestamp   TEXT NOT NULL
            )
        """)
        _conn.commit()
    return _conn

def log_feedback(
    thread_id: str,
    question: str,
    answer: str,
    thumbs_up: bool,
    comment: str = "",
    eval_score: float = 0.0,
):
    """Log user feedback to SQLite. Called from backend API on 👍/👎."""
    try:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO feedback 
               (thread_id, question, answer, thumbs_up, thumbs_down, user_comment, eval_score, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                thread_id, question[:500], answer[:1000],
                1 if thumbs_up else 0,
                0 if thumbs_up else 1,
                comment, eval_score,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        logger.info("feedback_logged", thumbs_up=thumbs_up, score=eval_score)
    except Exception as e:
        logger.error("feedback_log_failed", error=str(e))


def get_feedback_stats() -> dict:
    """Return aggregate feedback statistics for the dashboard."""
    try:
        conn = _get_conn()
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(thumbs_up) as positive,
                SUM(thumbs_down) as negative,
                AVG(eval_score) as avg_eval_score
            FROM feedback
        """)
        row = cursor.fetchone()
        
        total = row[0] or 0
        positive = row[1] or 0
        negative = row[2] or 0
        
        return {
            "total": total,
            "positive": positive,
            "negative": negative,
            "avg_eval_score": round(row[3] or 0.0, 2),
            "satisfaction_rate": round((positive) / max(total, 1) * 100, 1),
        }
    except Exception as e:
        logger.error("get_feedback_stats_failed", error=str(e))
        return {"total": 0, "positive": 0, "negative": 0, "avg_eval_score": 0.0, "satisfaction_rate": 0.0}
