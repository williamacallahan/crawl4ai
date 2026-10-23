"""Budget-prefetched archive HTML rendered offline with original URL identity."""
from contextlib import aclosing, asynccontextmanager

from archive_budget import fetch, is_archive_url
from crawl4ai.async_crawler_strategy import AsyncPlaywrightCrawlerStrategy


async def prefetch_archive(redis, config, urls):
    responses = {}
    for url in urls:
        if is_archive_url(url) and url not in responses:
            responses[url] = await fetch(redis, config, url, include_headers=True)
    return responses


class ArchiveHTMLStrategy(AsyncPlaywrightCrawlerStrategy):
    """Request-owned archive-only renderer; ordinary seeds use another crawler."""

    def __init__(self, responses, **kwargs):
        super().__init__(**kwargs)
        self.responses = responses
        self._offline_contexts = set()

    async def execute_hook(self, hook_type, *args, **kwargs):
        result = await super().execute_hook(hook_type, *args, **kwargs)
        if hook_type == "on_page_context_created":
            context = kwargs["context"]
            if context not in self._offline_contexts:
                # Replay assets are intentionally not fetched. Blocking foreign
                # assets too prevents their redirects back into archive.org.
                await context.set_offline(True)

                async def offline_socket(socket):
                    await socket.close()

                await context.route_web_socket("**/*", offline_socket)
                self._offline_contexts.add(context)
        return result

    async def crawl(self, url, config, **kwargs):
        fetched = self.responses[url]
        response = await super().crawl(
            "raw:" + fetched["body"],
            config=config.clone(base_url=fetched["url"]), **kwargs,
        )
        response.status_code = fetched["status"]
        response.response_headers = fetched["responseHeaders"]
        response.redirected_url = fetched["url"]
        return response


async def acquire_crawler(browser, responses, hooks=None):
    from crawler_pool import get_crawler, get_dedicated_crawler
    from api import _attach_declarative_hooks, _dispose_crawler

    if responses:
        strategy = ArchiveHTMLStrategy(responses, browser_config=browser)
        crawler = await get_dedicated_crawler(browser, crawler_strategy=strategy)
    elif hooks:
        crawler = await get_dedicated_crawler(browser)
    else:
        return await get_crawler(browser)
    try:
        if hooks:
            _attach_declarative_hooks(crawler, hooks)
        return crawler
    except BaseException:
        await _dispose_crawler(crawler)
        raise


@asynccontextmanager
async def owned_crawler(browser, responses, hooks):
    from api import _dispose_crawler

    crawler = await acquire_crawler(browser, responses, hooks)
    try:
        yield crawler
    finally:
        await _dispose_crawler(crawler)


async def grouped_results(urls, browser, run_config, responses, hooks,
                          dispatcher_factory):
    """Never hold two browser leases. Native dispatch selects original URLs."""
    groups = ([u for u in urls if u in responses],
              [u for u in urls if u not in responses])
    for group in groups:
        if not group:
            continue
        supplied = responses if group[0] in responses else {}
        async with owned_crawler(browser, supplied, hooks) as crawler:
            dispatcher = dispatcher_factory()
            try:
                result = await crawler.arun_many(
                    group, config=run_config, dispatcher=dispatcher,
                )
                if hasattr(result, "__aiter__"):
                    async with aclosing(result):
                        async for item in result:
                            yield item
                else:
                    for item in result:
                        yield item
            finally:
                close = getattr(dispatcher, "aclose", None)
                if close:
                    await close()


async def collect_results(results):
    return [item async for item in results]
