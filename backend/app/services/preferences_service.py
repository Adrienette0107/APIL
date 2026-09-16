from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def get_user_preferences(
    db: AsyncSession,
    user_id: str,
) -> dict:

    query = text("""
        SELECT
            language,
            level,
            response_length
        FROM user_preferences
        WHERE external_user_id = :user_id
    """)

    result = await db.execute(
        query,
        {"user_id": user_id},
    )

    row = result.fetchone()

    if not row:
        return {
            "language": "English",
            "level": "beginner",
            "response_length": "medium",
        }

    return {
        "language": row.language,
        "level": row.level,
        "response_length": row.response_length,
    }


async def save_user_preferences(
    db: AsyncSession,
    user_id: str,
    preferences: dict,
):
    query = text("""
        INSERT INTO user_preferences (
            external_user_id,
            language,
            level,
            response_length
        )
        VALUES (
            :user_id,
            :language,
            :level,
            :response_length
        )

        ON CONFLICT (external_user_id)
        DO UPDATE SET
            language = EXCLUDED.language,
            level = EXCLUDED.level,
            response_length = EXCLUDED.response_length,
            updated_at = NOW()
    """)

    await db.execute(
        query,
        {
            "user_id": user_id,
            "language": preferences.get(
                "language",
                "English",
            ),
            "level": preferences.get(
                "level",
                "beginner",
            ),
            "response_length": preferences.get(
                "response_length",
                "medium",
            ),
        },
    )
