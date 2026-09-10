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
    """Upload document(s) to pre-fill the brief via intelligent field extraction.

    Workflow:
      1. Validate file count and size.
      2. Route pure image uploads to upload_logo (existing behavior).
      3. Parse each document file using DocumentParser (PDF/DOCX/TXT/CSV).
      4. Run DocumentExtractor to sequentially extract fields from each document.
      5. Persist updated state to the database.
      6. Return a structured JSON response with extraction_report and snapshot.

    Returns:
        {
          "message":           str  — human-readable summary of what was extracted,
          "extraction_report": dict — per-document field accounting,
          "snapshot":          dict — full session snapshot (same format as SSE done event),
        }
    """
    from app.domain.document.document_extractor import DocumentExtractor
    from app.infrastructure.document_parser import parse_document, DocumentParseError

    if len(files) < 1 or len(files) > 5:
        raise HTTPException(status_code=400, detail="Please upload between 1 and 5 files.")

    # ── File size validation ─────────────────────────────────────────────────
    max_bytes = settings.document_max_file_size_mb * 1024 * 1024
    for file in files:
        # Read and immediately seek back so content is available below
        content_peek = await file.read()
        await file.seek(0)
        if len(content_peek) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"'{file.filename}' exceeds the maximum file size of "
                    f"{settings.document_max_file_size_mb} MB. "
                    "Please upload a smaller file."
                ),
            )

    # ── Route image-only uploads to the existing logo handler ────────────────
    image_files = [f for f in files if f.content_type and f.content_type.startswith("image/")]
    text_files = [f for f in files if not (f.content_type and f.content_type.startswith("image/"))]

    if image_files and not text_files:
        result = await upload_logo(session_id, orchestrator, image_files)
        state = await orchestrator._repo.get_session(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found after upload")
        active_profile = orchestrator._profile_provider(state)
        snapshot = state.to_snapshot(active_profile, settings.extraction_confidence_threshold)
        return {
            "message": result.get("message", "Images uploaded successfully."),
            "extraction_report": {},
            "snapshot": snapshot,
        }

    # ── Load session state ────────────────────────────────────────────────────
    state = await orchestrator._repo.get_session(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")

    active_profile = orchestrator._profile_provider(state)

    # ── Build shared extractor ────────────────────────────────────────────────
    extractor = DocumentExtractor(
        llm=orchestrator._llm,
        prompt_builder=orchestrator._prompt_builder,
    )

    # ── Process each document file sequentially ───────────────────────────────
    all_messages: list[str] = []
    combined_report: dict = {
        "documents_processed": [],
        "required_fields_extracted": [],
        "required_fields_not_found": [],
        "additional_fields_saved": [],
        "warnings": [],
    }

    for file in text_files:
        content = await file.read()

        # ── Parse document ───────────────────────────────────────────────────
        try:
            parsed_doc = parse_document(
                content=content,
                filename=file.filename or "uploaded_document",
                content_type=file.content_type,
            )
        except DocumentParseError as exc:
            logger.warning(
                "Document parse failed",
                extra={"session_id": session_id, "file_name": file.filename, "error": str(exc)},
            )
            raise HTTPException(status_code=400, detail=exc.user_message) from exc
        except Exception as exc:
            logger.error(
                "Document parse unexpected error",
                extra={"session_id": session_id, "file_name": file.filename},
                exc_info=exc,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    f"An unexpected error occurred while reading '{file.filename}'. "
                    "Please try again or upload a different file."
                ),
            ) from exc

        logger.info(
            "Document parsed for extraction",
            extra={
                "session_id": session_id,
                "file_name": file.filename,
                "parser": parsed_doc.parser_used,
                "chunks": len(parsed_doc.chunks),
                "chars": len(parsed_doc.raw_text),
            },
        )

        # ── Extract fields from the parsed document ──────────────────────────
        try:
            result = await extractor.extract(
                state=state,
                active_profile=active_profile,
                parsed_doc=parsed_doc,
            )
        except Exception as exc:
            logger.error(
                "Document extraction failed",
                extra={"session_id": session_id, "file_name": file.filename},
                exc_info=exc,
            )
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Field extraction from '{file.filename}' failed unexpectedly. "
                    "Please try again."
                ),
            ) from exc

        # ── Accumulate results across multiple documents ─────────────────────
        all_messages.append(result.extraction_summary)
        combined_report["documents_processed"].append(file.filename)
        combined_report["required_fields_extracted"].extend(result.fields_saved)
        combined_report["additional_fields_saved"].extend(result.additional_fields_saved)
        combined_report["warnings"].extend(result.warnings)

        logger.info(
            "Document extraction result",
            extra={
                "session_id": session_id,
                "file_name": file.filename,
                "fields_saved": result.fields_saved,
                "additional_fields_saved": result.additional_fields_saved,
                "missing_remaining": len(result.missing_required_fields),
                "chunks_processed": result.chunks_processed,
            },
        )

    # Deduplicate combined report entries
    combined_report["required_fields_extracted"] = list(dict.fromkeys(combined_report["required_fields_extracted"]))
    combined_report["additional_fields_saved"] = list(dict.fromkeys(combined_report["additional_fields_saved"]))

    # Compute final missing fields for the report
    final_missing = state.compute_missing_fields(active_profile, settings.extraction_confidence_threshold)
    combined_report["required_fields_not_found"] = [mf.field_code for mf in final_missing]

    # ── Persist state to DB ───────────────────────────────────────────────────
    try:
        await orchestrator._repo.save_session(state)
    except Exception as exc:
        logger.error(
            "upload_document: save_session failed after extraction",
            extra={"session_id": session_id},
            exc_info=exc,
        )
        raise HTTPException(
            status_code=500,
            detail="Extraction completed but the session could not be saved. Please try again.",
        ) from exc

    # ── Build final snapshot and response ────────────────────────────────────
    snapshot = state.to_snapshot(active_profile, settings.extraction_confidence_threshold)
    combined_message = "\n\n".join(all_messages) if all_messages else (
        "No supported document text was found. Please upload a PDF, DOCX, or TXT file."
    )

    return {
        "message": combined_message,
        "extraction_report": combined_report,
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

    # Bug Fix: Smart field routing for multi-upload support.
    # Priority 1: Check currently active missing fields for a file_upload field.
    # This handles the first upload (e.g. food photos → uploaded_files).
    target_field: str | None = None
    for f in missing:
        if getattr(f, "input_type", "") == "file_upload":
            target_field = f.field_code
            break

    # Priority 2: If no file_upload field is currently missing (e.g. the first
    # upload already satisfied uploaded_files, and brand_uploaded_files is now
    # active due to brand_identity_choice being set), scan ALL profile fields
    # for a file_upload field whose show_if dependency is satisfied.
    if target_field is None:
        for field_def in active_profile.required_fields:
            if field_def.input_type != "file_upload":
                continue
            if field_def.code in state.captured:
                continue  # already filled — skip
            # Check show_if dependency: if the dependency field is captured,
            # and the condition is met, this field is now active.
            if field_def.show_if:
                dep = state.captured.get(field_def.show_if.field_code)
                if not dep:
                    continue  # dependency not yet filled
                dep_vals = [v.strip().lower() for v in dep.value.split(",")]
                if field_def.show_if.in_:
                    cond_met = any(v in [x.lower() for x in field_def.show_if.in_] for v in dep_vals)
                elif field_def.show_if.not_in:
                    cond_met = not any(v in [x.lower() for x in field_def.show_if.not_in] for v in dep_vals)
                else:
                    cond_met = False
                if cond_met:
                    target_field = field_def.code
                    break
            else:
                # No condition — unconditional file_upload field, always active
                target_field = field_def.code
                break

    # Priority 3: Last resort fallback
    if target_field is None:
        target_field = "existing_assets"
        logger.warning(
            "upload_logo: no suitable file_upload field found — falling back to existing_assets",
            extra={"session_id": session_id},
        )

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
