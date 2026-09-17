import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.rate_limit import check_rate_limit
from backend.app.core.errors import ProviderExecutionError
from backend.app.core.security import verify_api_key
from backend.app.database import get_db
from backend.app.schemas.chat import ChatRequest, ChatResponse
from backend.app.services.conversation_service import (
    ensure_user,
    ensure_conversation,
    save_message,
)
from backend.app.services.provider_health import (
    ProviderHealthService,
)
provider_health_service = (
    ProviderHealthService()
)
from backend.app.services.apil_pipeline import process_chat
from backend.app.services.preferences_service import (
    save_user_preferences,
)

router = APIRouter(
    prefix="/v1",
    tags=["APIL Chat"],
)

logger = logging.getLogger("apil")


@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    http_request: Request,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(check_rate_limit),
    __: None = Depends(verify_api_key),
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
            request_id=request_id,
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
            processed_prompt=result["processed_prompt"],
            selected_provider=result["provider"],
            selected_model=result["model"],
            message=result["response"],
        )

    except PermissionError:
        await db.rollback()

        raise HTTPException(
            status_code=403,
            detail="Access to this conversation is not allowed.",
        )

    except ValueError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=400,
            detail={
                "code": "invalid_model_or_provider",
                "message": str(exc),
                "request_id": request_id,
            },
        ) from exc

    except ProviderExecutionError as exc:
        await db.rollback()

        raise HTTPException(
            status_code=exc.status_code,
            detail={
                "code": exc.code,
                "message": exc.message,
                "request_id": request_id,
            },
        ) from exc

    except Exception as exc:
        await db.rollback()

        logger.error(
            "APIL processing failed | request_id=%s | error_type=%s",
            request_id,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "message": "APIL processing failed.",
                "request_id": request_id,
            },
        ) from exc
@router.get("/providers/health")
async def provider_health():
    providers = (
        provider_health_service.get_all_health()
    )

    return {
        "status": "ok",
        "providers": [
            {
                "provider": item.provider,
                "enabled": item.enabled,
                "configured": item.configured,
                "circuit_state": item.circuit_state,
                "failure_count": item.failure_count,
                "priority": item.priority,
                "fallback_enabled": item.fallback_enabled,
                "default_model": item.default_model,
            }
            for item in providers
        ],
        "summary": (
            provider_health_service
            .get_summary()
        ),
    }