import os
import sqlite3
from datetime import datetime


DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")


def init_db():
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                result TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )


def save_result(result, confidence):
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO analysis_history (result, confidence, timestamp)
            VALUES (?, ?, ?)
            """,
            (result, confidence, datetime.now().isoformat(timespec="seconds")),
        )


def get_last_five_results():
    with sqlite3.connect(DB_PATH) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT result, confidence, timestamp
            FROM analysis_history
            ORDER BY id DESC
            LIMIT 5
            """
        ).fetchall()

    return [dict(row) for row in rows]
