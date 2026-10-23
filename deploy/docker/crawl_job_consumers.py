"""Retire the consumer records that exited crawl-job workers leave in Redis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from redis.exceptions import ResponseError

if TYPE_CHECKING:
    from crawl_job_queue import CrawlJobQueue

logger = logging.getLogger(__name__)

# Every worker process names its consumers after its host and pid, so each
# restart or rollout leaves the previous names behind in both groups forever.
# XGROUP DELCONSUMER discards the consumer's pending entries, so the PEL check
# and the delete share one script. 'idle' counts from the last attempted read
# or claim (Redis 7.2+), which an idle worker's blocking XREADGROUP and its
# outbox XAUTOCLAIM renew every read_block_ms; 'inactive' would not.
# Neither XINFO nor DELCONSUMER is a deny-OOM command, so this also runs when
# Redis is at maxmemory.
_REAP_IDLE_CONSUMERS_SCRIPT = """
-- crawl4ai:reap-idle-consumers
local removed = 0
for _, consumer in ipairs(redis.call('XINFO', 'CONSUMERS', KEYS[1], ARGV[1])) do
  if removed >= tonumber(ARGV[3]) then break end
  local fields = {}
  for index = 1, #consumer, 2 do
    fields[consumer[index]] = consumer[index + 1]
  end
  if fields['pending'] == 0 and fields['idle'] >= tonumber(ARGV[2]) then
    redis.call('XGROUP', 'DELCONSUMER', KEYS[1], ARGV[1], fields['name'])
    removed = removed + 1
  end
end
return removed
"""


async def reap_idle_consumers(queue: CrawlJobQueue) -> int:
    """Delete up to ``consumer_reap_batch`` empty, idle consumers per group.

    The idle floor is the attempt budget: a live consumer is never silent that
    long, and one that owns work keeps it because pending must be zero.
    """
    settings = queue.settings
    removed = 0
    for stream, group in ((settings.stream, settings.group),
                          (queue.outbox_stream, queue.outbox_group)):
        try:
            removed += int(await queue.redis.eval(
                _REAP_IDLE_CONSUMERS_SCRIPT, 1, stream, group,
                settings.max_attempt_seconds * 1000, settings.consumer_reap_batch,
            ))
        except ResponseError as error:
            # The outbox stream expires with its group when idle; a missing
            # group has no consumers left to retire.
            if "NOGROUP" not in str(error) and "no such key" not in str(error):
                raise
    if removed:
        logger.info("Retired %d idle crawl-job stream consumers", removed)
    return removed
