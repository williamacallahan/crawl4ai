"""Bounded, fenced terminal webhook delivery shared by crawl-job workers."""

import json
from uuid import uuid4

from redis.exceptions import ResponseError
from utils import decode_task_result
from webhook import DEFAULT_WEBHOOK_ATTEMPTS

OUTBOX_TTL_SECONDS = 86400

# Every event enters the PEL in the terminal transaction. Only its completing
# worker sends immediately; other workers reclaim after the crawl lease grace.
APPEND_OUTBOX_LUA = f"""
local function append_outbox(event, result, encoding, limit, group, consumer)
  if event == '' then return '' end
  local stream = KEYS[5]
  local attempts = stream .. ':attempts'
  -- Expiring an idle stream also removes its group, even in a live worker.
  redis.pcall('XGROUP', 'CREATE', stream, group .. ':webhooks', '0-0', 'MKSTREAM')
  local overflow = redis.call('XLEN', stream) - tonumber(limit) + 1
  if overflow > 0 then
    local evicted = redis.call('XRANGE', stream, '-', '+', 'COUNT', overflow)
    for _, row in ipairs(evicted) do
      redis.call('XACK', stream, group .. ':webhooks', row[1])
      redis.call('HDEL', attempts, row[1], row[1] .. ':operation')
    end
  end
  local id = redis.call('XADD', stream, 'MAXLEN', limit, '*',
    'event', event, 'result', result, 'result_encoding', encoding)
  redis.call('XCLAIM', stream, group .. ':webhooks', consumer, 0, id,
    'FORCE', 'JUSTID')
  redis.call('EXPIRE', stream, {OUTBOX_TTL_SECONDS})
  redis.call('EXPIRE', attempts, {OUTBOX_TTL_SECONDS})
  return id
end
"""


class CrawlJobOutbox:
    """Queue mixin; imports lease errors lazily to keep the queue owner canonical."""
    def _webhook_includes_result(self, webhook: dict) -> bool:
        """Whether this job's terminal webhook stores a second copy of its result."""
        settings = self.config.get("webhooks", {})
        url = webhook.get("webhook_url") or settings.get("default_url")
        return bool(settings.get("enabled", True) and url
                    and webhook.get("webhook_data_in_payload", settings.get("data_in_payload", False)))

    def _webhook_event(self, task_id: str, payload: dict, status: str, error: str | None):
        settings = self.config.get("webhooks", {})
        webhook = payload.get("webhook_config") or {}
        url = webhook.get("webhook_url") or settings.get("default_url")
        if not settings.get("enabled", True) or not url:
            return "", False
        include = self._webhook_includes_result(webhook)
        # Freeze effective delivery configuration: a restart must not change the recipient.
        webhook = {**webhook, "webhook_url": url, "webhook_data_in_payload": include}
        event = dict(task_id=task_id, task_type="crawl", status=status,
                     urls=payload["urls"], webhook_config=webhook, error=error,
                     event_id=f"crawl-terminal:{task_id}")
        return json.dumps(event), include

    def _outbox_events(self, messages):
        from crawl_job_queue import _as_text, _field
        events = []
        for stream_id, fields in messages:
            event = json.loads(_field(fields, "event"))
            result = fields.get(b"result", fields.get("result"))
            event["result"] = decode_task_result({"result": result,
                "result_encoding": _field(fields, "result_encoding")}, event["task_id"])
            events.append((_as_text(stream_id), event))
        return events

    async def read_outbox(self, consumer: str) -> list[tuple[str, dict]]:
        # No XREADGROUP '>': it would deliver freshly committed entries despite
        # the PEL reservation, bypassing the completing worker's grace period.
        try:
            claimed = await self.redis.xautoclaim(self.outbox_stream, self.outbox_group, consumer,
                self.settings.lease_seconds * 1000, getattr(self, "_outbox_cursor", "0-0"), count=1)
        except ResponseError as error:
            await self._recover_outbox_group(error)
            return []
        self._outbox_cursor = claimed[0]
        return self._outbox_events(claimed[1])

    async def pending_outbox(self, consumer: str) -> list[tuple[str, dict]]:
        try:
            pending = await self.redis.xpending_range(self.outbox_stream, self.outbox_group,
                "-", "+", self.settings.max_pending_jobs, consumername=consumer)
        except ResponseError as error:
            await self._recover_outbox_group(error)
            return []
        messages = []
        for row in pending:
            messages.extend(await self.redis.xrange(self.outbox_stream, row["message_id"], row["message_id"]))
        return self._outbox_events(messages)

    async def _recover_outbox_group(self, error: ResponseError) -> None:
        if "NOGROUP" not in str(error):
            raise error
        await self.ensure_group()
        self._outbox_cursor = "0-0"

    async def _outbox_owned(self, stream_id: str, consumer: str, action: str) -> int:
        script = """
local pending = redis.call('XPENDING', KEYS[1], ARGV[1], ARGV[3], ARGV[3], 1)
if #pending == 0 or pending[1][2] ~= ARGV[2] then return 0 end
-- A send that spent its retry budget is dropped like a delivered one: kept
-- in the PEL it would only queue ahead of crash-orphaned events.
if ARGV[4] == 'finish' or ARGV[4] == 'fail' then
  redis.call('XACK', KEYS[1], ARGV[1], ARGV[3])
  redis.call('XDEL', KEYS[1], ARGV[3])
  redis.call('HDEL', KEYS[2], ARGV[3], ARGV[3] .. ':operation')
  return 1
end
redis.call('XCLAIM', KEYS[1], ARGV[1], ARGV[2], 0, ARGV[3], 'IDLE', 0, 'JUSTID')
if ARGV[4] == 'reserve' then
  if redis.call('HGET', KEYS[2], ARGV[3] .. ':operation') == ARGV[5] then
    return tonumber(redis.call('HGET', KEYS[2], ARGV[3]))
  end
  local spent = tonumber(redis.call('HGET', KEYS[2], ARGV[3]) or '0')
  if spent >= tonumber(ARGV[6]) then return -1 end
  redis.call('HSET', KEYS[2], ARGV[3] .. ':operation', ARGV[5])
  local attempt = redis.call('HINCRBY', KEYS[2], ARGV[3], 1)
  redis.call('PEXPIREAT', KEYS[2], redis.call('PEXPIRETIME', KEYS[1]))
  return attempt
end
return 1
"""
        limit = self.config.get("webhooks", {}).get("retry", {}).get("max_attempts", DEFAULT_WEBHOOK_ATTEMPTS)
        return int(await self.redis.eval(script, 2, self.outbox_stream,
            f"{self.outbox_stream}:attempts", self.outbox_group, consumer, stream_id, action, uuid4().hex, limit))

    async def reserve_outbox_attempt(self, stream_id: str, consumer: str) -> int:
        return await self._outbox_owned(stream_id, consumer, "reserve")

    async def heartbeat_outbox(self, stream_id: str, consumer: str) -> None:
        if not await self._outbox_owned(stream_id, consumer, "heartbeat"):
            from crawl_job_queue import CrawlJobLeaseLost
            raise CrawlJobLeaseLost(f"webhook {stream_id} lease lost")

    async def fail_outbox(self, stream_id: str, consumer: str) -> None:
        """Drop an event whose retry budget is spent; the receiver can still poll."""
        if not await self._outbox_owned(stream_id, consumer, "fail"):
            from crawl_job_queue import CrawlJobLeaseLost
            raise CrawlJobLeaseLost(f"webhook {stream_id} lease lost")

    async def finish_outbox(self, stream_id: str, consumer: str) -> None:
        if not await self._outbox_owned(stream_id, consumer, "finish"):
            from crawl_job_queue import CrawlJobLeaseLost
            raise CrawlJobLeaseLost(f"webhook {stream_id} lease lost")

