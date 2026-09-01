#!/usr/bin/env python3
"""
Crawl4AI Docker server hooks - declarative REST examples.

The Docker server does NOT run user-supplied hook code. The old
``hooks.code`` map of Python strings was an exec()-based RCE surface and was
removed in 0.9.0. A request now picks from a fixed set of server-authored
actions with schema-validated parameters:

    block_resources, add_cookies, set_headers, scroll_to_bottom,
    wait_for_timeout

The payload shape (works from any language - it is plain JSON):

    {
      "urls": ["https://example.com"],
      "hooks": {
        "hooks": [
          {"action": "block_resources", "params": {"resource_types": ["image"]}},
          {"action": "scroll_to_bottom", "params": {"max_steps": 10, "delay_ms": 500}}
        ]
      }
    }

Notes:
- ``GET /hooks/info`` enumerates the actions and their parameter schemas.
- The server must run with ``CRAWL4AI_HOOKS_ENABLED=true``; otherwise a crawl
  carrying hooks is rejected with 403.
- There is no action for arbitrary JavaScript or DOM logic - by design. For
  that, use the in-process SDK (``AsyncWebCrawler`` +
  ``crawler_strategy.set_hook``), where your code is already trusted. See
  docs/examples/docker_client_hooks_example.py and deploy/docker/MIGRATION.md.

Requirements:
- Docker server running, e.g.:
    docker run -p 11235:11235 -e CRAWL4AI_HOOKS_ENABLED=true unclecode/crawl4ai:latest
"""

import json

import requests

DOCKER_URL = "http://localhost:11235"


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def example_1_discover_actions():
    """GET /hooks/info is the source of truth for actions and parameters."""
    print_section("1. Discover available actions (GET /hooks/info)")

    info = requests.get(f"{DOCKER_URL}/hooks/info", timeout=10).json()
    print(json.dumps(info["usage"], indent=2))
    for name, entry in info["available_actions"].items():
        print(f"\n- {name} ({entry['hook_point']}): {entry['description']}")
        print(json.dumps(entry["params_schema"], indent=2)[:400])


def example_2_performance_crawl():
    """Block heavy resources and scroll for lazy-loaded content."""
    print_section("2. Performance crawl: block_resources + scroll_to_bottom")

    payload = {
        "urls": ["https://httpbin.org/html"],
        "hooks": {
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
        "crawler_config": {"cache_mode": "bypass"},
    }

    response = requests.post(f"{DOCKER_URL}/crawl", json=payload, timeout=60)
    if response.status_code == 403:
        print("Hooks are disabled on this server (CRAWL4AI_HOOKS_ENABLED != true)")
        return
    response.raise_for_status()
    result = response.json()["results"][0]
    print(f"Success: {result['success']}, HTML: {len(result.get('html', ''))} chars")


def example_3_authentication_crawl():
    """Inject cookies and headers before navigation."""
    print_section("3. Authentication: add_cookies + set_headers")

    import base64

    credentials = base64.b64encode(b"user:passwd").decode("ascii")

    payload = {
        "urls": ["https://httpbin.org/basic-auth/user/passwd"],
        "hooks": {
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
                    "params": {"headers": {"Authorization": f"Basic {credentials}"}},
                },
            ]
        },
    }

    response = requests.post(f"{DOCKER_URL}/crawl", json=payload, timeout=60)
    if response.status_code == 403:
        print("Hooks are disabled on this server (CRAWL4AI_HOOKS_ENABLED != true)")
        return
    response.raise_for_status()
    result = response.json()["results"][0]
    authenticated = '"authenticated"' in result.get("html", "")
    print(f"Success: {result['success']}, basic auth worked: {authenticated}")


def example_4_wait_for_slow_pages():
    """Give slow or JS-heavy pages extra settle time with wait_for_timeout."""
    print_section("4. Waiting: wait_for_timeout")

    payload = {
        "urls": ["https://httpbin.org/html"],
        "hooks": {
            "hooks": [
                {"action": "wait_for_timeout", "params": {"timeout_ms": 1500}},
            ]
        },
    }

    response = requests.post(f"{DOCKER_URL}/crawl", json=payload, timeout=60)
    if response.status_code == 403:
        print("Hooks are disabled on this server (CRAWL4AI_HOOKS_ENABLED != true)")
        return
    response.raise_for_status()
    result = response.json()["results"][0]
    print(f"Success: {result['success']}")


def main():
    print("Crawl4AI Docker server - declarative hooks over REST")

    try:
        requests.get(f"{DOCKER_URL}/health", timeout=3).raise_for_status()
    except Exception:
        print(f"Server not reachable at {DOCKER_URL}.")
        print("Start it with: docker run -p 11235:11235 "
              "-e CRAWL4AI_HOOKS_ENABLED=true unclecode/crawl4ai:latest")
        return

    example_1_discover_actions()
    example_2_performance_crawl()
    example_3_authentication_crawl()
    example_4_wait_for_slow_pages()


if __name__ == "__main__":
    main()
