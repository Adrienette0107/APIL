from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.rate_limit import check_rate_limit
from backend.app.database import get_db
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.conversation_service import (
    ensure_user,
    ensure_conversation,
    save_message,
)
from backend.app.services.apil_pipeline import process_chat
from backend.app.services.preferences_service import (
    save_user_preferences,
)

router = APIRouter(
    prefix="/v1",
    tags=["APIL Chat"],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    http_request: Request,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(check_rate_limit),
):
    request_id = http_request.state.request_id

    try:

        await ensure_user(
            db=db,
            user_id=request.user_id,
        )

        await ensure_conversation(
            db=db,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )

        await save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="user",
            content=request.prompt,
        )

        preferences = (
            request.preferences.model_dump()
            if request.preferences
            else {}
        )

        if preferences:
            await save_user_preferences(
                db=db,
                user_id=request.user_id,
                preferences=preferences,
            )

        result = await process_chat(
            db=db,
            prompt=request.prompt,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            model=request.model,
            preferences=preferences,
        )

        await save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="assistant",
            content=result["response"],
        )

        await db.commit()

        return ChatResponse(
            request_id=request_id,
            status="success",
            original_prompt=request.prompt,
            selected_model=result["model"],
            message=result["response"],
        )

    except PermissionError:
        await db.rollback()

        raise HTTPException(
            status_code=403,
            detail="Access to this conversation is not allowed.",
        )

    except Exception:
        await db.rollback()

        raise HTTPException(
            status_code=500,
            detail={
                "message": "APIL processing failed.",
                "request_id": request_id,
            },
        )