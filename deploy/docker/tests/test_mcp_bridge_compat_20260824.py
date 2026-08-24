"""Compatibility coverage for the locked MCP SDK resource contracts."""

import asyncio

import httpx
import mcp.types as mcp_types
import mcp_bridge
from fastapi import FastAPI


def test_resources_and_templates_use_mcp_uris_and_read_handlers(monkeypatch):
    """Registered resource callbacks expose and accept the MCP SDK's URI fields."""

    original_server = mcp_bridge.Server

    class CapturingServer(original_server):
        instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            type(self).instance = self

    monkeypatch.setattr(mcp_bridge, "Server", CapturingServer)
    app = FastAPI()

    @app.get("/snapshot")
    @mcp_bridge.mcp_resource("snapshot")
    async def snapshot():
        return {"ok": True}

    @app.get("/items/{item_id:int}")
    @mcp_bridge.mcp_template("item")
    async def item(item_id: int):
        return {"item_id": item_id}

    mcp_bridge.attach_mcp(app, base_url="http://127.0.0.1:9999")
    server = CapturingServer.instance
    assert server is not None

    async def get_schema():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await client.get("/mcp/schema")

    schema_response = asyncio.run(get_schema())
    assert schema_response.status_code == 200
    assert schema_response.json()["resources"][0]["uri"] == "crawl4ai://resources/snapshot"
    assert schema_response.json()["resource_templates"][0]["uriTemplate"] == (
        "crawl4ai://resources/items/{item_id}"
    )

    resource_result = asyncio.run(
        server.request_handlers[mcp_types.ListResourcesRequest](
            mcp_types.ListResourcesRequest()
        )
    ).root
    resource = resource_result.resources[0]
    assert resource.name == "snapshot"
    assert str(resource.uri) == "crawl4ai://resources/snapshot"
    assert resource.mimeType == "application/json"

    template_result = asyncio.run(
        server.request_handlers[mcp_types.ListResourceTemplatesRequest](
            mcp_types.ListResourceTemplatesRequest()
        )
    ).root
    template = template_result.resourceTemplates[0]
    assert template.name == "item"
    assert template.uriTemplate == "crawl4ai://resources/items/{item_id}"
    assert template.mimeType == "application/json"

    resource_read = asyncio.run(
        server.request_handlers[mcp_types.ReadResourceRequest](
            mcp_types.ReadResourceRequest(
                params={"uri": "crawl4ai://resources/snapshot"}
            )
        )
    ).root
    assert resource_read.contents[0].text == '{"ok": true}'
    assert resource_read.contents[0].mimeType == "application/json"

    template_read = asyncio.run(
        server.request_handlers[mcp_types.ReadResourceRequest](
            mcp_types.ReadResourceRequest(
                params={"uri": "crawl4ai://resources/items/42"}
            )
        )
    ).root
    assert template_read.contents[0].text == '{"item_id": 42}'
