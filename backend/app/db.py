"""SQLite persistence for run history. Kept lightweight (no ORM) since this is a
single-table append log, not a relational domain model — pulling in SQLAlchemy
for one table would be over-engineering."""
import os
import sqlite3
import json
from app.config import config


def save_run(result: dict):
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT, as_of TEXT, created_at TEXT, json_result TEXT
        )""")
        conn.execute(
            "INSERT INTO runs (ticker, as_of, created_at, json_result) VALUES (?,?,?,?)",
            (result.get("ticker"), result.get("as_of"),
             result.get("generated_at"), json.dumps(result)),
        )
        conn.commit()
    finally:
        conn.close()
