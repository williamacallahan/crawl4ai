from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from utils import FilterType

CrawlResultField = Literal[
    "url",
    "redirected_url",
    "success",
    "error_message",
    "status_code",
    "html",
    "markdown",
    "media",
    "links",
    "metadata",
    "response_headers",
]
CRAWL_RESULT_FIELDS = get_args(CrawlResultField)


def validate_crawl_result_fields(result_fields):
    required = {"success", "error_message"}
    if result_fields is not None and not required.issubset(result_fields):
        raise ValueError("result_fields must include 'success' and 'error_message'")
    return result_fields


class CrawlRequest(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=100)
    browser_config: dict | None = Field(default_factory=dict)
    crawler_config: dict | None = Field(default_factory=dict)
    crawler_configs: list[dict] | None = Field(
        default=None,
        description=(
            "List of per-URL CrawlerRunConfig dicts for arun_many(). "
            "When provided, each config can include a 'url_matcher' pattern "
            "to match against specific URLs. Takes precedence over crawler_config."
        ),
    )
    result_fields: list[CrawlResultField] | None = None

    @field_validator("result_fields")
    @classmethod
    def require_status_result_fields(cls, result_fields):
        return validate_crawl_result_fields(result_fields)


class HookSpec(BaseModel):
    """A single declarative hook: a fixed action plus schema-validated params.

    Arbitrary Python (the old `code` map) is no longer accepted - it was an
    exec()-based RCE surface. Available actions are enumerated by GET /hooks/info
    and validated server-side by hook_registry.py.
    """

    action: str = Field(..., description="One of the registered hook actions")
    params: dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )


class HookConfig(BaseModel):
    """Configuration for declarative hooks."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hooks": [
                    {
                        "action": "block_resources",
                        "params": {"resource_types": ["image", "font"]},
                    },
                    {
                        "action": "scroll_to_bottom",
                        "params": {"max_steps": 10, "delay_ms": 500},
                    },
                ],
            }
        }
    )

    hooks: list[HookSpec] = Field(
        default_factory=list,
        max_length=10,
        description="Declarative hook specs (action + params), max 10",
    )


class CrawlRequestWithHooks(CrawlRequest):
    """Extended crawl request with hooks support"""

    hooks: HookConfig | None = Field(
        default=None, description="Optional user-provided hook functions"
    )


class MarkdownRequest(BaseModel):
    """Request body for the /md endpoint."""

    url: str = Field(..., description="Absolute http/https URL to fetch")
    f: FilterType = Field(
        FilterType.FIT, description="Content‑filter strategy: fit, raw, bm25, or llm"
    )
    q: str | None = Field(None, description="Query string used by BM25/LLM filters")
    c: str | None = Field("0", description="Cache‑bust / revision counter")
    provider: str | None = Field(
        None, description="LLM provider override (e.g., 'anthropic/claude-3-opus')"
    )
    temperature: float | None = Field(
        None, description="LLM temperature override (0.0-2.0)"
    )
    # base_url removed: a request-supplied LLM endpoint was a credential-exfil
    # vector. The endpoint is derived server-side from the provider name.


class RawCode(BaseModel):
    code: str


class HTMLRequest(BaseModel):
    url: str


class ScreenshotRequest(BaseModel):
    url: str
    screenshot_wait_for: float | None = 2
    wait_for_images: bool | None = False
    # output_path removed: callers never name a filesystem path (it was an
    # arbitrary-write -> RCE vector). The server writes to the sandboxed
    # artifact store and returns an opaque artifact_id.


class PDFRequest(BaseModel):
    url: str
    # output_path removed (see ScreenshotRequest).


class JSEndpointRequest(BaseModel):
    url: str
    scripts: list[str] = Field(
        ..., description="List of separated JavaScript snippets to execute"
    )


class WebhookConfig(BaseModel):
    """Configuration for webhook notifications."""

    webhook_url: HttpUrl
    webhook_data_in_payload: bool = False
    webhook_headers: dict[str, str] | None = None

    @field_validator("webhook_headers")
    @classmethod
    def _validate_headers(cls, v):
        # Reject unsafe outbound headers early (422). Mirrors
        # webhook.sanitize_webhook_headers; kept inline to avoid an import cycle.
        if not v:
            return v
        from webhook import sanitize_webhook_headers

        return sanitize_webhook_headers(v)


class WebhookPayload(BaseModel):
    """Payload sent to webhook endpoints."""

    task_id: str
    task_type: str  # "crawl", "llm_extraction", etc.
    status: str  # "completed" or "failed"
    timestamp: str  # ISO 8601 format
    urls: list[str]
    error: str | None = None
    data: dict | None = None  # Included only if webhook_data_in_payload=True
