from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.errors import ProviderExecutionError
from backend.app.core.config import settings
from backend.app.core.rate_limit import check_rate_limit
from backend.app.core.security import verify_api_key
from backend.app.database import get_db
from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)
from backend.app.services.apil_pipeline import (
    process_chat,
)
from backend.app.services.conversation_service import (
    ensure_conversation,
    ensure_user,
    save_message,
)
from backend.app.services.preferences_service import (
    save_user_preferences,
)
from backend.app.services.provider_health import (
    ProviderHealthService,
)
from backend.app.services.response_sanitizer import (
    sanitize_model_output,
)


router = APIRouter(
    prefix="/v1",
    tags=["APIL Chat"],
)

logger = logging.getLogger("apil")

provider_health_service = ProviderHealthService()


# ----------------------------------------------------------------------
# Main APIL chat endpoint
# ----------------------------------------------------------------------

@router.post(
    "/chat",
    response_model=ChatResponse,
)
async def chat(
    http_request: Request,
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    _: None = Depends(verify_api_key),
    __: None = Depends(check_rate_limit),
):
    request_id = getattr(
        http_request.state,
        "request_id",
        None,
    )

    try:

        # --------------------------------------------------------------
        # 1. Ensure user exists
        # --------------------------------------------------------------

        await ensure_user(
            db=db,
            user_id=request.user_id,
        )

        # --------------------------------------------------------------
        # 2. Ensure conversation exists
        # --------------------------------------------------------------

        await ensure_conversation(
            db=db,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
        )

        # --------------------------------------------------------------
        # 3. Save user message
        # --------------------------------------------------------------

        await save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="user",
            content=request.prompt,
        )

        # --------------------------------------------------------------
        # 4. Extract request preferences
        # --------------------------------------------------------------

        preferences = (
            request.preferences.model_dump(
                exclude_none=True
            )
            if request.preferences
            else {}
        )

        # --------------------------------------------------------------
        # 5. Persist updated preferences
        # --------------------------------------------------------------

        if preferences:

            await save_user_preferences(
                db=db,
                user_id=request.user_id,
                preferences=preferences,
            )

        # --------------------------------------------------------------
        # 6. Execute canonical APIL pipeline
        #
        # /v1/chat
        #     ↓
        # process_chat()
        #     ↓
        # prompt optimization
        #     ↓
        # model routing
        #     ↓
        # provider fallback
        #     ↓
        # output verification
        #     ↓
        # output evaluation
        #     ↓
        # conditional improvement
        # --------------------------------------------------------------

        result = await process_chat(
            db=db,
            prompt=request.prompt,
            user_id=request.user_id,
            conversation_id=request.conversation_id,
            model=request.model,
            preferences=preferences,
            request_id=request_id,
        )

        # --------------------------------------------------------------
        # 7. Save final assistant response
        # --------------------------------------------------------------

        await save_message(
            db=db,
            conversation_id=request.conversation_id,
            role="assistant",
            content=result["response"],
        )

        # --------------------------------------------------------------
        # 8. Commit database transaction
        # --------------------------------------------------------------

        await db.commit()

        # --------------------------------------------------------------
        # 9. Return clean Swagger response
        # --------------------------------------------------------------

        return ChatResponse(
            request_id=request_id,
            status="success",

            original_prompt=(
                request.prompt
            ),

            optimized_prompt=(
                result.get(
                    "optimized_prompt",
                    request.prompt,
                )
            ),

            prompt_dna=(
                result.get(
                    "prompt_dna",
                    {},
                )
            ),

            processed_prompt=(
                result.get(
                    "processed_prompt",
                    result.get(
                        "optimized_prompt",
                        request.prompt,
                    ),
                )
            ),

            selected_provider=(
                result.get(
                    "provider",
                    "unknown",
                )
            ),

            selected_model=(
                result.get(
                    "model",
                    "unknown",
                )
            ),

            response=(
                result.get(
                    "response",
                    "",
                )
            ),

            response_evaluation=(
                result.get(
                    "response_evaluation"
                )
            ),

            improvement_applied=bool(
                result.get(
                    "improvement_applied",
                    False,
                )
            ),

            message=(
                result.get(
                    "response",
                    "",
                )
            ),

            provider_call_count=int(
                result.get(
                    "provider_call_count",
                    1,
                )
                or 1
            ),

            timing=(
                result.get(
                    "timing"
                )
            ),

            improvement_attempted=bool(
                result.get(
                    "improvement_attempted",
                    False,
                )
            ),

            improvement_failure_reason=(
                result.get(
                    "improvement_error"
                )
            ),

            raw_provider_response=(
                result.get("raw_provider_response")
                if settings.APIL_ENABLE_DEBUG_METADATA
                else None
            ),

            raw_response=(
                result.get("raw_response")
                if settings.APIL_ENABLE_DEBUG_METADATA
                else None
            ),

            final_response=result.get("final_response"),
            verification=result.get("verification"),
            final_quality_gate=result.get("final_quality_gate"),
            improvement_attempts=int(result.get("improvement_attempts", 0) or 0),
            improvement_needed=bool(result.get("improvement_needed", False)),
            improvement_error=result.get("improvement_error"),
            provider_attempts=int(result.get("provider_attempts", 1) or 1),
            preferences=result.get("preferences"),
            analysis=result.get("analysis"),
        )

    # ------------------------------------------------------------------
    # Permission errors
    # ------------------------------------------------------------------

    except PermissionError:

        await db.rollback()

        raise HTTPException(
            status_code=403,
            detail={
                "code": "conversation_access_denied",
                "message": (
                    "Access to this conversation "
                    "is not allowed."
                ),
                "request_id": request_id,
            },
        )

    # ------------------------------------------------------------------
    # Validation / model selection errors
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Provider execution errors
    # ------------------------------------------------------------------

    except ProviderExecutionError as exc:

        await db.rollback()

        raise HTTPException(
            status_code=(
                exc.status_code
                or 502
            ),
            detail={
                "code": exc.code,
                "message": exc.message,
                "provider": exc.provider,
                "category": exc.category,
                "retryable": exc.retryable,
                "attempts": exc.attempts,
                "request_id": request_id,
            },
        ) from exc

    # ------------------------------------------------------------------
    # Unexpected errors
    # ------------------------------------------------------------------

    except Exception as exc:

        await db.rollback()

        logger.exception(
            "APIL processing failed | "
            "request_id=%s | error_type=%s",
            request_id,
            type(exc).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail={
                "code": "apil_processing_error",
                "message": (
                    "APIL processing failed."
                ),
                "request_id": request_id,
            },
        ) from exc


# ----------------------------------------------------------------------
# Diagnostic endpoint
# ----------------------------------------------------------------------

@router.post("/test")
async def diagnostic_test(
    payload: dict,
):
    """
    Lightweight diagnostic endpoint.

    This endpoint does NOT execute a GenAI provider.
    It only normalizes an already-produced APIL result.
    """

    original_prompt = (
        payload.get(
            "original_prompt",
            "",
        )
    )

    final_response = (
        payload.get(
            "final_response"
        )
        or payload.get(
            "response"
        )
        or payload.get(
            "raw_provider_response"
        )
        or ""
    )

    raw_provider_response = (
        payload.get(
            "raw_provider_response"
        )
        or final_response
    )

    return {
        "status": "success",

        "original_prompt": (
            original_prompt
        ),

        "prompt_dna": (
            payload.get(
                "prompt_dna"
            )
            or {}
        ),

        "optimized_prompt": (
            payload.get(
                "optimized_prompt",
                original_prompt,
            )
        ),

        "processed_prompt": (
            payload.get(
                "processed_prompt",
                payload.get(
                    "optimized_prompt",
                    original_prompt,
                ),
            )
        ),

        "selected_provider": (
            payload.get(
                "selected_provider",
                payload.get(
                    "provider",
                    "unknown",
                ),
            )
        ),

        "selected_model": (
            payload.get(
                "selected_model",
                payload.get(
                    "model",
                    "unknown",
                ),
            )
        ),

        "provider_call_count": int(
            payload.get(
                "provider_call_count",
                1,
            )
            or 1
        ),

        "raw_provider_response": (
            sanitize_model_output(
                raw_provider_response
            )
        ),

        "response_evaluation": (
            payload.get(
                "response_evaluation"
            )
            or {}
        ),

        "improvement_needed": bool(
            payload.get(
                "improvement_needed",
                False,
            )
        ),

        "improvement_attempts": int(
            payload.get(
                "improvement_attempts",
                0,
            )
            or 0
        ),

        "improvement_attempted": bool(
            payload.get(
                "improvement_attempted",
                False,
            )
        ),

        "improvement_applied": bool(
            payload.get(
                "improvement_applied",
                False,
            )
        ),

        "improvement_failure_reason": (
            payload.get(
                "improvement_failure_reason"
            )
            or payload.get(
                "improvement_error"
            )
        ),

        "improved_response": (
            sanitize_model_output(
                payload.get(
                    "improved_response",
                    final_response,
                )
            )
        ),

        "final_quality_gate": (
            payload.get(
                "final_quality_gate"
            )
            or {
                "passed": True
            }
        ),

        "final_response": (
            sanitize_model_output(final_response)
        ),

        "timing": (
            payload.get(
                "timing"
            )
            or {
                "total_ms": 0
            }
        ),
    }


# ----------------------------------------------------------------------
# Provider health
# ----------------------------------------------------------------------

@router.get(
    "/providers/health"
)
async def provider_health():

    providers = (
        provider_health_service
        .get_all_health()
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