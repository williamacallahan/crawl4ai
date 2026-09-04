# monitor_routes.py - Monitor API endpoints
import asyncio
import logging

from auth import require_admin
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from monitor import get_monitor
from pydantic import BaseModel
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/monitor", tags=["monitor"])


@router.get("/health")
async def get_health():
    """Get current system health snapshot."""
    try:
        monitor = get_monitor()
        return await monitor.get_health_summary()
    except Exception as e:
        logger.error(f"Error getting health: {e}")
        raise HTTPException(500, str(e))


@router.get("/requests")
async def get_requests(status: str = "all", limit: int = 50):
    """Get active and completed requests.

    Args:
        status: Filter by 'active', 'completed', 'success', 'error', or 'all'
        limit: Max number of completed requests to return (default 50)
    """
    # Input validation
    if status not in ["all", "active", "completed", "success", "error"]:
        raise HTTPException(400, f"Invalid status: {status}. Must be one of: all, active, completed, success, error")
    if limit < 1 or limit > 1000:
        raise HTTPException(400, f"Invalid limit: {limit}. Must be between 1 and 1000")

    try:
        monitor = get_monitor()

        if status == "active":
            return {"active": monitor.get_active_requests(), "completed": []}
        elif status == "completed":
            return {"active": [], "completed": monitor.get_completed_requests(limit)}
        elif status in ["success", "error"]:
            return {"active": [], "completed": monitor.get_completed_requests(limit, status)}
        else:  # "all"
            return {
                "active": monitor.get_active_requests(),
                "completed": monitor.get_completed_requests(limit)
            }
    except Exception as e:
        logger.error(f"Error getting requests: {e}")
        raise HTTPException(500, str(e))


@router.get("/browsers")
async def get_browsers():
    """Get detailed browser pool information."""
    try:
        monitor = get_monitor()
        browsers = await monitor.get_browser_list()

        # Calculate summary stats
        total_browsers = len(browsers)
        total_memory = sum(b["memory_mb"] for b in browsers)

        # Calculate reuse rate from recent requests
        recent = monitor.get_completed_requests(100)
        pool_hits = sum(1 for r in recent if r.get("pool_hit", False))
        reuse_rate = (pool_hits / len(recent) * 100) if recent else 0

        return {
            "browsers": browsers,
            "summary": {
                "total_count": total_browsers,
                "total_memory_mb": total_memory,
                "reuse_rate_percent": round(reuse_rate, 1)
            }
        }
    except Exception as e:
        logger.error(f"Error getting browsers: {e}")
        raise HTTPException(500, str(e))


@router.get("/endpoints/stats")
async def get_endpoint_stats():
    """Get aggregated endpoint statistics."""
    try:
        monitor = get_monitor()
        return monitor.get_endpoint_stats_summary()
    except Exception as e:
        logger.error(f"Error getting endpoint stats: {e}")
        raise HTTPException(500, str(e))


@router.get("/timeline")
async def get_timeline(metric: str = "memory", window: str = "5m"):
    """Get timeline data for charts.

    Args:
        metric: 'memory', 'requests', or 'browsers'
        window: Time window (only '5m' supported for now)
    """
    # Input validation
    if metric not in ["memory", "requests", "browsers"]:
        raise HTTPException(400, f"Invalid metric: {metric}. Must be one of: memory, requests, browsers")
    if window != "5m":
        raise HTTPException(400, f"Invalid window: {window}. Only '5m' is currently supported")

    try:
        monitor = get_monitor()
        return monitor.get_timeline_data(metric, window)
    except Exception as e:
        logger.error(f"Error getting timeline: {e}")
        raise HTTPException(500, str(e))


@router.get("/logs/janitor")
async def get_janitor_log(limit: int = 100):
    """Get recent janitor cleanup events."""
    # Input validation
    if limit < 1 or limit > 1000:
        raise HTTPException(400, f"Invalid limit: {limit}. Must be between 1 and 1000")

    try:
        monitor = get_monitor()
        return {"events": monitor.get_janitor_log(limit)}
    except Exception as e:
        logger.error(f"Error getting janitor log: {e}")
        raise HTTPException(500, str(e))


@router.get("/logs/errors")
async def get_errors_log(limit: int = 100):
    """Get recent errors."""
    # Input validation
    if limit < 1 or limit > 1000:
        raise HTTPException(400, f"Invalid limit: {limit}. Must be between 1 and 1000")

    try:
        monitor = get_monitor()
        return {"errors": monitor.get_errors_log(limit)}
    except Exception as e:
        logger.error(f"Error getting errors log: {e}")
        raise HTTPException(500, str(e))


# ========== Control Actions ==========

class KillBrowserRequest(BaseModel):
    sig: str


@router.post("/actions/cleanup", dependencies=[Depends(require_admin)])
async def force_cleanup():
    """Force immediate janitor cleanup (kills idle cold pool browsers)."""
    try:
        from crawler_pool import retire_pool_crawlers

        retired = await retire_pool_crawlers(cold_only=True)
        for sig, _pool_type in retired:
            logger.info(f"🧹 Force cleanup: closing cold browser (sig={sig[:8]})")

        monitor = get_monitor()
        await monitor.track_janitor_event(
            "force_cleanup", "manual", {"killed": len(retired)}
        )

        return {"success": True, "killed_browsers": len(retired)}
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")
        raise HTTPException(500, str(e))


@router.post("/actions/kill_browser", dependencies=[Depends(require_admin)])
async def kill_browser(req: KillBrowserRequest):
    """Kill a specific browser by signature (hot or cold only).

    Args:
        sig: Browser config signature (first 8 chars)
    """
    try:
        from crawler_pool import (
            DEFAULT_CONFIG_SIG,
            retire_pool_crawlers,
        )

        if DEFAULT_CONFIG_SIG and DEFAULT_CONFIG_SIG.startswith(req.sig):
            raise HTTPException(403, "Cannot kill permanent browser. Use restart instead.")

        retired = await retire_pool_crawlers(req.sig)
        if not retired:
            raise HTTPException(404, f"Browser with sig={req.sig} not found")
        target_sig, pool_type = retired[0]

        monitor = get_monitor()
        active_count = len(monitor.get_active_requests())
        if active_count > 0:
            logger.warning(f"Killing browser {target_sig[:8]} while {active_count} requests are active - may cause failures")

        logger.info(f"🔪 Killed {pool_type} browser (sig={target_sig[:8]})")

        monitor = get_monitor()
        await monitor.track_janitor_event("kill_browser", target_sig, {"pool": pool_type, "manual": True})

        return {"success": True, "killed_sig": target_sig[:8], "pool_type": pool_type}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error killing browser: {e}")
        raise HTTPException(500, str(e))


@router.post("/actions/restart_browser", dependencies=[Depends(require_admin)])
async def restart_browser(req: KillBrowserRequest):
    """Restart a browser (kill + recreate). Works for permanent too.

    Args:
        sig: Browser config signature (first 8 chars), or "permanent"
    """
    try:
        from crawler_pool import (
            DEFAULT_CONFIG_SIG,
            init_permanent,
            retire_pool_crawlers,
        )
        # Handle permanent browser restart
        if req.sig == "permanent" or (DEFAULT_CONFIG_SIG and DEFAULT_CONFIG_SIG.startswith(req.sig)):
            from server import get_default_browser_config

            await init_permanent(get_default_browser_config(), force=True)

            logger.info("🔄 Restarted permanent browser")
            return {"success": True, "restarted": "permanent"}

        # Handle hot/cold browser restart
        retired = await retire_pool_crawlers(req.sig)
        if not retired:
            raise HTTPException(404, f"Browser with sig={req.sig} not found")
        target_sig, pool_type = retired[0]

        logger.info(f"🔄 Restarted {pool_type} browser (sig={target_sig[:8]})")

        monitor = get_monitor()
        await monitor.track_janitor_event("restart_browser", target_sig, {"pool": pool_type})

        return {"success": True, "restarted_sig": target_sig[:8], "note": "Browser will be recreated on next request"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restarting browser: {e}")
        raise HTTPException(500, str(e))


@router.post("/stats/reset", dependencies=[Depends(require_admin)])
async def reset_stats():
    """Reset today's endpoint counters."""
    try:
        monitor = get_monitor()
        monitor.endpoint_stats.clear()
        await monitor._persist_endpoint_stats()

        return {"success": True, "message": "Endpoint stats reset"}
    except Exception as e:
        logger.error(f"Error resetting stats: {e}")
        raise HTTPException(500, str(e))


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates.

    Sends updates every 2 seconds with:
    - Health stats
    - Active/completed requests
    - Browser pool status
    - Timeline data
    """
    # Auth is enforced by the AuthGateMiddleware (outermost ASGI layer), which
    # validates Authorization or the browser WebSocket bearer subprotocol and
    # closes 4401 before we are reached. No bespoke check here.
    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            try:
                # Gather all monitoring data
                monitor = get_monitor()

                data = {
                    "timestamp": asyncio.get_event_loop().time(),
                    "health": await monitor.get_health_summary(),
                    "requests": {
                        "active": monitor.get_active_requests(),
                        "completed": monitor.get_completed_requests(limit=10)
                    },
                    "browsers": await monitor.get_browser_list(),
                    "timeline": {
                        "memory": monitor.get_timeline_data("memory", "5m"),
                        "requests": monitor.get_timeline_data("requests", "5m"),
                        "browsers": monitor.get_timeline_data("browsers", "5m")
                    },
                    "janitor": monitor.get_janitor_log(limit=10),
                    "errors": monitor.get_errors_log(limit=10)
                }

                # Send update to client
                await websocket.send_json(data)

                # Wait 2 seconds before next update
                await asyncio.sleep(2)

            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
                break
            except Exception as e:
                # A send on an already-closed socket raises RuntimeError, not
                # WebSocketDisconnect, so the retry re-raised it every 2s and
                # spammed a traceback for the whole stop-grace window on every
                # rollout (observed 2026-08-31, haiku-18). A closed socket is
                # terminal: stop instead of retrying.
                if websocket.application_state is not WebSocketState.CONNECTED:
                    logger.info("WebSocket closed during update: %s", e)
                    break
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await asyncio.sleep(2)  # Continue trying

    except Exception as e:
        logger.error(f"WebSocket connection error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed")
