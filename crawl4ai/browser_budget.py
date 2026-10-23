"""One task-local deadline for all browser work belonging to a crawl."""

import asyncio
from contextvars import ContextVar


page_deadline: ContextVar[float | None] = ContextVar("page_deadline", default=None)


async def within_page_budget(operation):
    """Bound renderer operations, including evaluate/content which have no timeout."""
    deadline = page_deadline.get()
    if deadline is None:
        return await operation
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    return await asyncio.wait_for(operation, timeout=remaining)
