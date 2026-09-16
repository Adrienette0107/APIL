from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def ensure_user(
    db: AsyncSession,
    user_id: str,
):
    query = text("""
        INSERT INTO users (external_user_id)
        VALUES (:user_id)
        ON CONFLICT (external_user_id)
        DO NOTHING
    """)

    await db.execute(
        query,
        {"user_id": user_id},
    )


async def ensure_conversation(
    db: AsyncSession,
    user_id: str,
    conversation_id: str,
):
    check_query = text("""
        SELECT external_user_id
        FROM conversations
        WHERE conversation_id = :conversation_id
    """)

    result = await db.execute(
        check_query,
        {"conversation_id": conversation_id},
    )

    existing = result.fetchone()

    if existing is None:
        insert_query = text("""
            INSERT INTO conversations (
                conversation_id,
                external_user_id
            )
            VALUES (
                :conversation_id,
                :user_id
            )
        """)

        await db.execute(
            insert_query,
            {
                "conversation_id": conversation_id,
                "user_id": user_id,
            },
        )

        return

    if existing.external_user_id != user_id:
        raise PermissionError(
            "Conversation does not belong to this user."
        )


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
):
    query = text("""
        INSERT INTO messages (
            conversation_id,
            role,
            content
        )
        VALUES (
            :conversation_id,
            :role,
            :content
        )
    """)

    await db.execute(
        query,
        {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
        },
    )


async def get_conversation_history(
    db: AsyncSession,
    user_id: str,
    conversation_id: str,
    limit: int = 20,
):
    query = text("""
        SELECT
            m.role,
            m.content
        FROM messages m
        INNER JOIN conversations c
            ON c.conversation_id = m.conversation_id
        WHERE
            m.conversation_id = :conversation_id
            AND c.external_user_id = :user_id
        ORDER BY
            m.created_at ASC,
            m.id ASC
        LIMIT :limit
    """)

    result = await db.execute(
        query,
        {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "limit": limit,
        },
    )

    return [
        {
            "role": row.role,
            "content": row.content,
        }
        for row in result.fetchall()
    ]