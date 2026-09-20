import importlib
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from packaging.requirements import InvalidRequirement, Requirement

from fastapi import HTTPException

ROOT = Path(__file__).resolve().parent.parent


def test_default_docker_dependencies_include_pypdf():
    lines = (ROOT / "deploy" / "docker" / "requirements.txt").read_text().splitlines()
    names = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith(("#", "-")):
            continue
        try:
            names.add(Requirement(line).name)
        except InvalidRequirement:
            continue

    assert "pypdf" in names


@pytest.mark.asyncio
@pytest.mark.parametrize("handler_name", ["handle_crawl_request", "handle_stream_crawl_request"])
async def test_server_rejects_pdf_scraping_before_crawler_admission(monkeypatch, handler_name):
    docker_dir = ROOT / "deploy" / "docker"
    monkeypatch.syspath_prepend(str(docker_dir))

    api = importlib.import_module("api")
    crawler_pool = importlib.import_module("crawler_pool")
    utils = importlib.import_module("utils")
    get_crawler = AsyncMock()
    monkeypatch.setattr(crawler_pool, "get_crawler", get_crawler)
    monkeypatch.setattr(api, "_normalize_and_validate_seeds", AsyncMock(side_effect=lambda urls: urls))
    monkeypatch.setattr(api, "_track_request_start", AsyncMock(return_value="pdf-rejection"))
    monkeypatch.setattr(api, "_close_aborted_request", AsyncMock())

    # SDK PDF downloads do not use the server's pre-connect pinning proxy.
    # The public API therefore keeps this strategy outside its type allowlist.
    crawler_config = {
        "type": "CrawlerRunConfig",
        "params": {
            "scraping_strategy": {
                "type": "PDFContentScrapingStrategy",
                "params": {"extract_images": False, "batch_size": 8},
            },
        },
    }
    with pytest.raises(HTTPException) as rejected:
        await getattr(api, handler_name)(
            urls=["https://example.com/document.pdf"],
            browser_config={},
            crawler_config=crawler_config,
            config=utils.load_config(),
        )

    assert rejected.value.status_code == 400
    assert "PDFContentScrapingStrategy" in rejected.value.detail
    get_crawler.assert_not_awaited()
