from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.conversation_service import (
    get_conversation_history,
)


async def build_context(
    db: AsyncSession,
    user_id: str,
    conversation_id: str,
    limit: int = 20,
) -> list[dict]:

    return await get_conversation_history(
        db=db,
        user_id=user_id,
        conversation_id=conversation_id,
        limit=limit,
    )
