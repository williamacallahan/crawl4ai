#!/usr/bin/env python3
"""
Declarative hooks with the Crawl4AI Docker client.

The Docker server does NOT execute user-supplied hook code (the exec()-based
hook system was removed in 0.9.0 to prevent RCE). Instead, a crawl request
selects from a fixed set of server-authored actions with schema-validated
parameters:

    block_resources, add_cookies, set_headers, scroll_to_bottom,
    wait_for_timeout

Discover the actions and their parameter schemas with ``GET /hooks/info``.
The server must be started with ``CRAWL4AI_HOOKS_ENABLED=true``; otherwise a
crawl that carries hooks is rejected with 403. See deploy/docker/MIGRATION.md.

Callable Python hooks still exist - but only in-process, with the SDK
(``AsyncWebCrawler``). The last example shows that path.

Requirements:
- Docker server running, e.g.:
    docker run -p 11235:11235 -e CRAWL4AI_HOOKS_ENABLED=true unclecode/crawl4ai:latest
- pip install crawl4ai
"""

import asyncio

import requests

from crawl4ai import Crawl4aiDockerClient

API_BASE_URL = "http://localhost:11235"


def show_available_actions():
    """GET /hooks/info enumerates every action and its parameter schema."""
    print("=" * 70)
    print("Available declarative hook actions (GET /hooks/info)")
    print("=" * 70)

    info = requests.get(f"{API_BASE_URL}/hooks/info", timeout=10).json()
    for name, entry in info["available_actions"].items():
        print(f"  - {name} ({entry['hook_point']}): {entry['description']}")
    print(f"\nRequest shape: {info['usage']['shape']}")
    print(f"Max hooks per request: {info['usage']['max_hooks']}")


async def crawl_with_performance_hooks():
    """Block heavy resources and scroll for lazy content."""
    print("\n" + "=" * 70)
    print("Performance: block resources + scroll to bottom")
    print("=" * 70)

    async with Crawl4aiDockerClient(base_url=API_BASE_URL, verbose=False) as client:
        result = await client.crawl(
            ["https://httpbin.org/html"],
            hooks={
                "hooks": [
                    {
                        "action": "block_resources",
                        "params": {"resource_types": ["image", "font", "media"]},
                    },
                    {
                        "action": "scroll_to_bottom",
                        "params": {"max_steps": 10, "delay_ms": 300},
                    },
                ]
            },
        )

    print(f"Success: {result.success}, HTML: {len(result.html)} chars")


async def crawl_with_authentication_hooks():
    """Inject auth state with add_cookies and set_headers."""
    print("\n" + "=" * 70)
    print("Authentication: cookies + headers")
    print("=" * 70)

    import base64

    credentials = base64.b64encode(b"user:passwd").decode("ascii")

    async with Crawl4aiDockerClient(base_url=API_BASE_URL, verbose=False) as client:
        result = await client.crawl(
            ["https://httpbin.org/basic-auth/user/passwd"],
            hooks={
                "hooks": [
                    {
                        "action": "add_cookies",
                        "params": {
                            "cookies": [
                                {
                                    "name": "session_id",
                                    "value": "example-session-token",
                                    "domain": ".httpbin.org",
                                    "path": "/",
                                }
                            ]
                        },
                    },
                    {
                        "action": "set_headers",
                        "params": {
                            "headers": {
                                "Authorization": f"Basic {credentials}",
                                "X-API-Key": "test-key-123",
                            }
                        },
                    },
                ]
            },
        )

    if result.success and '"authenticated"' in result.html:
        print("Basic auth succeeded")
    else:
        print(f"Success: {result.success} (auth status unclear)")


async def crawl_multiple_urls_with_hooks():
    """The same hook list applies to every URL in the request."""
    print("\n" + "=" * 70)
    print("Multi-URL crawl with shared hooks")
    print("=" * 70)

    hooks = {
        "hooks": [
            {"action": "block_resources", "params": {"resource_types": ["image"]}},
            {"action": "wait_for_timeout", "params": {"timeout_ms": 500}},
        ]
    }

    async with Crawl4aiDockerClient(base_url=API_BASE_URL, verbose=False) as client:
        results = await client.crawl(
            [
                "https://httpbin.org/html",
                "https://httpbin.org/json",
                "https://httpbin.org/xml",
            ],
            hooks=hooks,
        )

    for result in results if isinstance(results, list) else [results]:
        print(f"  {'ok ' if result.success else 'ERR'} {result.url}")


async def sdk_callable_hooks():
    """Arbitrary hook code runs only in-process with AsyncWebCrawler.

    There is no declarative action for custom JavaScript, DOM inspection, or
    conditional logic - by design. When you need that power, run the crawler
    in your own process where the code is already trusted. Requires a local
    Playwright browser install (``playwright install chromium``).
    """
    print("\n" + "=" * 70)
    print("In-process SDK: callable hooks (not the Docker server)")
    print("=" * 70)

    from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig

    async def before_goto(page, context, url, **kwargs):
        print(f"  [hook] navigating to {url}")
        await page.set_extra_http_headers({"X-Custom-Header": "sdk-hook"})
        return page

    async with AsyncWebCrawler() as crawler:
        crawler.crawler_strategy.set_hook("before_goto", before_goto)
        result = await crawler.arun(
            "https://httpbin.org/html",
            config=CrawlerRunConfig(cache_mode=CacheMode.BYPASS),
        )
        print(f"Success: {result.success}, HTML: {len(result.html)} chars")


async def main():
    print("Crawl4AI Docker client - declarative hooks")
    show_available_actions()
    await crawl_with_performance_hooks()
    await crawl_with_authentication_hooks()
    await crawl_multiple_urls_with_hooks()
    try:
        await sdk_callable_hooks()
    except Exception as exc:  # needs a local browser install
        print(f"SDK example skipped: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
