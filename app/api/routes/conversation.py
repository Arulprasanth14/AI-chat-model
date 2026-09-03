"""
app/api/routes/conversation.py
────────────────────────────────
Conversation SSE endpoint.

POST /conversation/message
  - Accepts: {session_id?: str, user_message: str, vertical?: str, template_key?: str}
  - Returns: Server-Sent Events stream
    - Regular events: data: {"chunk": "..."} \\n\\n
    - Final event:   data: {"done": true, "snapshot": {...}} \\n\\n

GET /conversation/session/{session_id}
  - Returns: current session snapshot

GET /conversation/session/{session_id}/brief
  - Returns: deterministic brief summary as markdown string

This mirrors the contract of the V2 NestJS service but uses Python SSE.
The ``sse-starlette`` package handles SSE headers and formatting.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, AsyncIterator

import cloudinary
import cloudinary.uploader

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import get_orchestrator
from app.core.config import settings
from app.domain.conversation.brief_renderer import render_brief
from app.domain.conversation.orchestrator import ConversationOrchestrator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversation", tags=["conversation"])


# ── Request/Response models ────────────────────────────────────────────────────

class MessageRequest(BaseModel):
    """Request body for POST /conversation/message."""

    session_id: str | None = Field(
        default=None,
        description="Existing session UUID. Omit to start a new session.",
    )
    user_message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's message text.",
    )
    vertical: str | None = Field(
        default=None,
        description=(
            "Pre-selected vertical from the UI selection screen. "
            "E.g. 'restaurant', 'realestate', 'ecommerce'. "
            "Only meaningful on the first message (session_id=null). "
            "When provided, bypasses the auto-detection heuristic."
        ),
    )
    template_key: str | None = Field(
        default=None,
        description=(
            "Pre-selected field-set template key from the UI selection screen. "
            "E.g. 'restaurant_cafe_static_post'. "
            "Only meaningful on the first message (session_id=null)."
        ),
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.post(
    "/message",
    summary="Send a message and receive a streaming SSE response",
    response_class=StreamingResponse,
)
async def post_message(
    body: MessageRequest,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
) -> StreamingResponse:
    """Process a user message and stream the AI response via SSE.

    The stream emits two event types:
      1. ``{"chunk": "..."}`` — incremental text tokens of the assistant's reply.
      2. ``{"done": true, "snapshot": {...}}`` — final event with full session state.

    The ``snapshot`` object contains:
      - session_id, profile_id, status
      - extracted_answers: {field_code: {value, confidence, turn_index}}
      - missing_fields: [{field_code, description}]
      - model_believes_complete, is_complete, turn_count

    Args:
        body:         Request with optional session_id and user_message.
        orchestrator: Injected ConversationOrchestrator.

    Returns:
        StreamingResponse with text/event-stream content type.
    """
    logger.info(
        "Conversation message received",
        extra={
            "session_id": body.session_id,
            "message_length": len(body.user_message),
        },
    )

    async def event_stream() -> AsyncIterator[str]:
        async for event in orchestrator.process_turn(
            session_id=body.session_id,
            user_message=body.user_message,
            vertical=body.vertical,
            template_key=body.template_key,
        ):
            yield event

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
            "Connection": "keep-alive",
        },
    )


@router.get(
    "/session/{session_id}",
    summary="Get current session snapshot",
)
async def get_session(
    session_id: str,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """Retrieve the current state snapshot for an existing session.

    Useful for the testing UI to reload state on page refresh.

    Args:
        session_id: UUID of the session to retrieve.

    Returns:
        Session snapshot dict (same format as the SSE done event snapshot).
    """
    state = await orchestrator._repo.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    return state.to_snapshot(orchestrator._profile_provider(state), 0.7)


@router.get(
    "/session/{session_id}/brief",
    summary="Get the deterministic brief summary for a session",
    response_model=dict,
)
async def get_brief(
    session_id: str,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """Return the deterministic rendered brief summary for the given session.

    Uses the same renderer as the orchestrator — guaranteed identical output
    for the same captured fields. Suitable for the UI's "View Full Brief" button.

    Args:
        session_id: UUID of the session.

    Returns:
        {"brief": "<markdown string>", "is_complete": bool, "session_id": str}
    """
    state = await orchestrator._repo.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    active_profile = orchestrator._profile_provider(state)
    is_complete = state.is_complete(active_profile, settings.extraction_confidence_threshold)

    field_set_yaml_path: Path | None = None
    if orchestrator._field_sets_root and state.resolved_vertical and state.resolved_template_key:
        field_set_yaml_path = (
            orchestrator._field_sets_root
            / state.resolved_vertical
            / f"{state.resolved_template_key}.yaml"
        )

    brief_md = render_brief(
        captured=state.captured,
        field_set_yaml_path=field_set_yaml_path,
        profile=active_profile,
        include_confidence=False,
    )

    return {
        "session_id": session_id,
        "is_complete": is_complete,
        "brief": brief_md,
    }


@router.post(
    "/session/{session_id}/document",
    summary="Upload document(s) and extract fields",
    response_model=dict,
)
async def upload_document(
    session_id: str,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
    files: list[UploadFile] = File(...),
) -> dict:
    """Upload document(s) to pre-fill the brief.
    
    Reuses the exact same orchestrator turn logic to prevent divergence.
    """
    if len(files) < 1 or len(files) > 5:
        raise HTTPException(status_code=400, detail="Please upload between 1 and 5 files.")

    image_files = []
    text_files = []
    
    for file in files:
        if file.content_type and file.content_type.startswith("image/"):
            image_files.append(file)
        else:
            text_files.append(file)
            
    if image_files and not text_files:
        # All files are images, process as asset uploads
        result = await upload_logo(session_id, orchestrator, image_files)
        
        # In case the session could not be fetched after upload
        state = await orchestrator._repo.get_session(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found after upload")
            
        active_profile = orchestrator._profile_provider(state)
        snapshot = state.to_snapshot(active_profile, settings.extraction_confidence_threshold)
        return {
            "message": result.get("message", "Images uploaded successfully."),
            "snapshot": snapshot
        }

    all_text = []
    for file in text_files:
        content = await file.read()
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            text = content.decode("latin-1")
            
        # Strip null bytes to prevent PostgreSQL JSONB crashes (UntranslatableCharacterError)
        text = text.replace("\x00", "")
        all_text.append(f"--- Document: {file.filename} ---\n{text}")

    combined_text = "\n\n".join(all_text)

    # We construct a user message that forces the LLM to extract fields from the document
    # and also explicitly asks it to mention any additional info found.
    prompt = (
        f"I have uploaded document(s) for this brief. Please extract all relevant information from them "
        f"into the correct fields. If there is any important information in the document that doesn't "
        f"match a known required field, please surface it to me by summarizing it in your conversational reply "
        f"as 'additional info found'.\n\nDocument Content:\n{combined_text}"
    )

    import json
    
    stream = orchestrator.process_turn(
        session_id=session_id,
        user_message=prompt,
    )
    
    assistant_msg = ""
    snapshot = None
    
    async for event in stream:
        if event.startswith("data: "):
            try:
                data = json.loads(event[6:])
                if "chunk" in data:
                    assistant_msg += data["chunk"]
                if "done" in data:
                    snapshot = data["snapshot"]
            except Exception as e:
                logger.error(f"Error parsing SSE event in document upload: {e}")

    return {
        "message": assistant_msg,
        "snapshot": snapshot,
    }


# ── Bug 3 / Bug 13 fix: Logo upload with hard field confirmation ───────────────

@router.post(
    "/session/{session_id}/logo",
    summary="Upload multiple files with confirmed field write",
    response_model=dict,
)
async def upload_logo(
    session_id: str,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
    files: list[UploadFile] = File(...),
) -> dict:
    """Upload logo(s) or brand assets and write directly to the target field.

    Bypasses LLM extraction — the upload IS the write. Written at confidence 1.0.
    Dynamically finds the missing file_upload field to satisfy.
    """
    if len(files) < 1 or len(files) > 5:
        raise HTTPException(status_code=400, detail="Please upload between 1 and 5 files.")

    state = await orchestrator._repo.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    active_profile = orchestrator._profile_provider(state)
    missing = state.compute_missing_fields(active_profile, settings.extraction_confidence_threshold)
    
    target_field = "existing_assets"
    for f in missing:
        if getattr(f, "input_type", "") == "file_upload":
            target_field = f.field_code
            break

    if not settings.cloudinary_url:
        return {
            "status": "failed",
            "field_code": target_field,
            "message": "Cloudinary is not configured. Please add CLOUDINARY_URL to your .env file.",
        }

    secure_urls = []
    filenames = []
    
    import os
    os.environ["CLOUDINARY_URL"] = settings.cloudinary_url
    cloudinary.reset_config()

    for file in files:
        filename = file.filename or "uploaded_asset"
        filenames.append(filename)
        try:
            content = await file.read()
            upload_result = cloudinary.uploader.upload(
                content,
                resource_type="auto",
                public_id=f"picasso_fusion/{session_id}/{filename.split('.')[0]}",
            )
            secure_urls.append(upload_result.get("secure_url"))
        except Exception as exc:
            logger.error(f"Cloudinary upload failed for {filename}", exc_info=exc)
            return {
                "status": "failed",
                "field_code": target_field,
                "message": f"Cloudinary upload failed for {filename}: {exc}",
            }

    value = ", ".join(secure_urls)
    result = state.handle_save_text_field(
        field_code=target_field,
        value=value,
        confidence=1.0,
        profile=active_profile,
        confidence_threshold=0.0,
    )

    if result.status == "saved":
        try:
            await orchestrator._repo.save_session(state)
        except Exception as exc:
            logger.error(
                "Logo upload: save_session failed",
                extra={"session_id": session_id, "uploaded_files": filenames},
                exc_info=exc,
            )
            return {
                "status": "failed",
                "field_code": target_field,
                "message": f"Upload received but failed to save: {exc}",
            }

        logger.info(
            "Logo/asset uploaded and confirmed",
            extra={"session_id": session_id, "uploaded_files": filenames},
        )
        snapshot = state.to_snapshot(active_profile, settings.extraction_confidence_threshold)
        file_count = len(filenames)
        return {
            "status": "confirmed",
            "field_code": target_field,
            "filename": ", ".join(filenames),
            "message": f"✅ {file_count} file(s) uploaded and saved.",
            "snapshot": snapshot,
        }

    return {
        "status": "failed",
        "field_code": target_field,
        "message": f"Upload rejected: {result.reason}",
    }


# ── Cluster D: FieldSpec-driven UI endpoints ───────────────────────────────────

@router.get(
    "/session/{session_id}/next_field_spec",
    summary="Get the FieldSpec for the next missing required field",
    response_model=dict,
)
async def get_next_field_spec(
    session_id: str,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
) -> dict:
    """Return the FieldSpec for the first missing required field.

    Frontend uses this to render the correct input control instead of always
    defaulting to a free-text box. Fixes Bugs 11, 12, 14, 15.

    Returns: {"next_field": FieldSpec | null}
    """
    from app.domain.conversation.field_spec_registry import get_field_spec

    state = await orchestrator._repo.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    active_profile = orchestrator._profile_provider(state)
    missing = state.compute_missing_fields(active_profile, settings.extraction_confidence_threshold)

    if not missing:
        return {"next_field": None}

    field_def = active_profile.get_field_by_code(missing[0].field_code)
    if field_def is None:
        return {"next_field": None}

    spec = get_field_spec(field_def)
    return {"next_field": spec.model_dump()}


@router.post(
    "/session/{session_id}/direct_field_write",
    summary="Persist a field value selected via structured UI (no LLM re-extraction)",
    response_model=dict,
)
async def direct_field_write(
    session_id: str,
    orchestrator: Annotated[ConversationOrchestrator, Depends(get_orchestrator)],
    body: dict,
) -> dict:
    """Write a field value directly from a UI selection, bypassing LLM extraction.

    Values written here are at confidence 1.0 — they will never be overwritten by
    lower-confidence LLM inferences. Fixes Bug 16 (repeated manual entry).

    Body: {"field_code": str, "value": str}
    """
    state = await orchestrator._repo.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    field_code = body.get("field_code", "")
    value = body.get("value", "")

    if not field_code or value is None or value == "":
        raise HTTPException(status_code=422, detail="field_code and value are required")

    active_profile = orchestrator._profile_provider(state)
    field_def = active_profile.get_field_by_code(field_code)

    if field_def is None:
        raise HTTPException(status_code=422, detail=f"Unknown field_code: {field_code!r}")

    if field_def.enum_values:
        result = state.handle_save_enum_field(
            field_code=field_code,
            value=str(value),
            confidence=1.0,
            profile=active_profile,
            confidence_threshold=0.0,
        )
    else:
        result = state.handle_save_text_field(
            field_code=field_code,
            value=str(value),
            confidence=1.0,
            profile=active_profile,
            confidence_threshold=0.0,
        )

    if result.status == "saved":
        try:
            await orchestrator._repo.save_session(state)
        except Exception as exc:
            logger.error(
                "direct_field_write: save_session failed",
                extra={"session_id": session_id, "field_code": field_code},
                exc_info=exc,
            )

    # Bug 5 fix: Return snapshot so the frontend can update its state panel immediately
    # after a chip click without waiting for the subsequent Phase B SSE stream.
    snapshot = state.to_snapshot(active_profile, settings.extraction_confidence_threshold)
    return {
        "status": result.status,
        "field_code": field_code,
        "value": value,
        "reason": result.reason,
        "snapshot": snapshot,
    }
