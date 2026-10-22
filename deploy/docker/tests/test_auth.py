import os
import sys
from datetime import timedelta

import httpx
import jwt
import pytest

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import auth


# These two pin the fork's GehirnInc-jwt -> PyJWT migration. Upstream 0.9.x made
# the same migration and replaced verify_token(HTTPAuthorizationCredentials),
# which raised HTTPException, with decode_token(str), which raises
# jwt.InvalidTokenError - enforcement now lives in AuthGateMiddleware. Same
# intent, retargeted at the surviving owner.
def test_access_token_round_trip_uses_locked_pyjwt():
    token = auth.create_access_token({"sub": "test@example.com"})

    assert auth.decode_token(token)["sub"] == "test@example.com"


def test_expired_access_token_is_rejected():
    token = auth.create_access_token(
        {"sub": "test@example.com"},
        expires_delta=timedelta(seconds=-1),
    )

    with pytest.raises(jwt.ExpiredSignatureError):
        auth.decode_token(token)


@pytest.mark.asyncio
async def test_comma_separated_api_tokens_each_authenticate(
    server_module, monkeypatch
):
    monkeypatch.setenv("CRAWL4AI_API_TOKEN", " old-token , new-token ,, ")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server_module.app),
        base_url="http://test",
    ) as client:
        old = await client.get("/schema", headers={"Authorization": "Bearer old-token"})
        new = await client.get("/schema", headers={"Authorization": "Bearer new-token"})
        unknown = await client.get(
            "/schema", headers={"Authorization": "Bearer other-token"}
        )
        joined = await client.get(
            "/schema", headers={"Authorization": "Bearer old-token,new-token"}
        )

    assert old.status_code == 200
    assert new.status_code == 200
    assert unknown.status_code == 401
    assert joined.status_code == 401
    assert server_module._internal_service_auth_headers() == {
        "Authorization": "Bearer old-token"
    }
