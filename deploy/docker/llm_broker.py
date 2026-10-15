"""
llm_broker.py - server-side LLM provider resolution.

The credential-exfil gadget (reported by Geo): a request could pass its own
`base_url` (and/or `provider`), and the server would happily send the
configured provider API key to that attacker-controlled endpoint, leaking
*every* provider key the server holds.

The fix: a request may select a provider BY NAME only (and only from an
allowlisted family). The `base_url` and `api_token` are ALWAYS derived
server-side from config/environment and are never taken from the request, so a
key can never be redirected to an attacker host.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Optional


class LLMWorkload(str, Enum):
    PAGE_QUESTION = "page-question"
    JOB_EXTRACTION = "job-extraction"
    MARKDOWN_FILTER = "markdown-filter"

    @property
    def tier(self) -> str:
        return "batch" if self is LLMWorkload.JOB_EXTRACTION else "production-a"


class LLMProviderNotAllowed(ValueError):
    """The requested LLM provider is not in the server's allowlist (-> HTTP 400)."""


def _family(provider: Optional[str]) -> str:
    return (provider or "").split("/")[0].lower()


def allowed_provider_families(config: Dict) -> set:
    """Provider families a request may select.

    Sourced from config.llm.allowed_providers plus the default provider's
    family. Empty set => unrestricted family selection (base_url/key are still
    server-only, so the exfil path stays closed); set a non-empty
    allowed_providers to lock it down further.
    """
    cfg = config.get("llm", {}) or {}
    fams = {_family(p) for p in (cfg.get("allowed_providers") or [])}
    if cfg.get("provider"):
        fams.add(_family(cfg["provider"]))
    return {f for f in fams if f}


def resolve_llm(
    config: Dict, requested_provider: Optional[str] = None, *, workload: LLMWorkload
) -> Dict:
    """Resolve the LLM call parameters fully server-side.

    A request-supplied base_url/api_token is intentionally NOT a parameter here:
    callers pass only the provider *name*. Returns {provider, base_url,
    api_token, temperature}, all server-derived.
    """
    from utils import get_llm_api_key, get_llm_base_url, get_llm_temperature

    default = config["llm"]["provider"]
    provider = requested_provider or default

    fams = allowed_provider_families(config)
    if fams and _family(provider) not in fams:
        raise LLMProviderNotAllowed("LLM provider not allowed")

    return {
        "provider": provider,
        "base_url": get_llm_base_url(config, provider),   # canonical, never from request
        "api_token": get_llm_api_key(config, provider),   # server credential
        "temperature": get_llm_temperature(config, provider),
        "extra_args": {
            "timeout": config["llm"].get("request_timeout_seconds", 300),
            "num_retries": config["llm"].get("request_retries", 0),
            # The configured effort belongs to the default model. A named model
            # override inherits its own gateway reasoning contract.
            **({"reasoning_effort": config["llm"]["reasoning_effort"],
                "allowed_openai_params": ["reasoning_effort"]}
               if provider == default and "reasoning_effort" in config["llm"] else {}),
            "max_tokens": config["llm"].get("max_output_tokens", 4096),
            "extra_headers": {
                "X-User-ID": f"crawl4ai:{workload.value}:{'batch' if workload.tier == 'batch' else 'live'}",
                "X-Tier": workload.tier,
            },
        },
    }
