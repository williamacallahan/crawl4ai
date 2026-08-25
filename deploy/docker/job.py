"""
Job endpoints (enqueue + poll) for long-running LL\u200bM extraction and raw crawl.
Relies on the existing Redis task helpers in api.py
"""

import asyncio
from typing import Annotated, Any

from api import (
    handle_crawl_job,
    handle_llm_request,
    handle_task_status,
)
from auth import get_principal
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field, HttpUrl, field_validator
from schemas import (
    CRAWL_RESULT_FIELDS,
    CrawlRequest,
    CrawlResultField,
    WebhookConfig,
    validate_crawl_result_fields,
)

# ------------- dependency placeholders -------------
_redis: Any = None  # injected from server.py before the router serves requests
_config: Any = None


def _token_dep():
    return None


# public router
router = APIRouter()


def _principal_dep(request: Request) -> dict | None:
    """The principal the AuthGateMiddleware already validated for this request."""
    return get_principal(request)


def _owner_of(principal: dict | None) -> str | None:
    return principal.get("sub") if principal else None


def _is_admin(principal: dict | None) -> bool:
    return bool(principal) and principal.get("scope") == "admin"


# === init hook called by server.py =========================================
def init_job_router(redis, config, token_dep) -> APIRouter:
    """Inject shared singletons and return the router for mounting."""
    global _redis, _config, _token_dep
    _redis, _config, _token_dep = redis, config, token_dep
    return router


# ---------- payload models --------------------------------------------------
class LlmJobPayload(BaseModel):
    url: HttpUrl
    q: str
    schema_: str | None = Field(default=None, alias="schema")
    cache: bool = False
    provider: str | None = None
    webhook_config: WebhookConfig | None = None
    temperature: float | None = None
    # base_url removed: server-derived LLM endpoint only (key-exfil vector).


class CrawlJobPayload(BaseModel):
    urls: list[HttpUrl]
    browser_config: dict = Field(default_factory=dict)
    crawler_config: dict = Field(default_factory=dict)
    result_fields: list[CrawlResultField] = Field(
        default_factory=lambda: list(CRAWL_RESULT_FIELDS)
    )
    webhook_config: WebhookConfig | None = None

    @field_validator("urls", mode="before")
    @classmethod
    def enforce_canonical_url_list_bounds(cls, urls):
        """Delegate list cardinality to the canonical synchronous crawl model."""
        return CrawlRequest.model_validate({"urls": urls}).urls

    @field_validator("result_fields")
    @classmethod
    def require_status_result_fields(cls, result_fields):
        return validate_crawl_result_fields(result_fields)


# ---------- LL​M job ---------------------------------------------------------
@router.post("/llm/job", status_code=202)
async def llm_job_enqueue(
    payload: LlmJobPayload,
    background_tasks: BackgroundTasks,
    request: Request,
    _td: Annotated[dict | None, Depends(_principal_dep)],
):
    webhook_config = None
    if payload.webhook_config:
        from utils import validate_webhook_url

        try:
            await asyncio.to_thread(
                validate_webhook_url,
                str(payload.webhook_config.webhook_url),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        webhook_config = payload.webhook_config.model_dump(mode="json")

    return await handle_llm_request(
        _redis,
        background_tasks,
        request,
        str(payload.url),
        query=payload.q,
        schema=payload.schema_,
        cache="1" if payload.cache else "0",
        config=_config,
        provider=payload.provider,
        webhook_config=webhook_config,
        temperature=payload.temperature,
        requester=_owner_of(_td),
        is_admin=_is_admin(_td),
    )


@router.get("/llm/job/{task_id}")
async def llm_job_status(
    request: Request,
    task_id: str,
    _td: Annotated[dict | None, Depends(_principal_dep)],
):
    return await handle_task_status(
        _redis,
        task_id,
        base_url=str(request.base_url),
        requester=_owner_of(_td),
        is_admin=_is_admin(_td),
    )


# ---------- CRAWL job -------------------------------------------------------
@router.post("/crawl/job", status_code=202)
async def crawl_job_enqueue(
    payload: CrawlJobPayload,
    background_tasks: BackgroundTasks,
    _td: Annotated[dict | None, Depends(_principal_dep)],
):
    webhook_config = None
    if payload.webhook_config:
        from utils import validate_webhook_url

        try:
            await asyncio.to_thread(
                validate_webhook_url,
                str(payload.webhook_config.webhook_url),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        webhook_config = payload.webhook_config.model_dump(mode="json")

    return await handle_crawl_job(
        _redis,
        [str(u) for u in payload.urls],
        payload.browser_config,
        payload.crawler_config,
        config=_config,
        result_fields=(
            [str(field) for field in payload.result_fields]
            if payload.result_fields is not None
            else None
        ),
        webhook_config=webhook_config,
        owner=_owner_of(_td),
    )


@router.get("/crawl/job/{task_id}")
async def crawl_job_status(
    request: Request,
    task_id: str,
    _td: Annotated[dict | None, Depends(_principal_dep)],
):
    return await handle_task_status(
        _redis,
        task_id,
        base_url=str(request.base_url),
        requester=_owner_of(_td),
        is_admin=_is_admin(_td),
    )
