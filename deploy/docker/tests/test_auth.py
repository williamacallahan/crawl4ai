import os
import sys
from datetime import timedelta

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
