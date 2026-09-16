import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent.parent / "apil.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def create_tables():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def save_message(user_id, conversation_id, role, message):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO conversations
        (user_id, conversation_id, role, message)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, conversation_id, role, message)
    )

    connection.commit()
    connection.close()


def get_messages(conversation_id):
    connection = get_connection()

    rows = connection.execute(
        """
        SELECT role, message
        FROM conversations
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,)
    ).fetchall()

    connection.close()

    return rows