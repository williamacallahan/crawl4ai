import os
import sys
from datetime import timedelta

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

DOCKER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DOCKER_DIR not in sys.path:
    sys.path.insert(0, DOCKER_DIR)

import auth


def credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def test_access_token_round_trip_uses_locked_pyjwt():
    token = auth.create_access_token({"sub": "test@example.com"})

    assert auth.verify_token(credentials(token))["sub"] == "test@example.com"


def test_expired_access_token_is_rejected():
    token = auth.create_access_token(
        {"sub": "test@example.com"},
        expires_delta=timedelta(seconds=-1),
    )

    with pytest.raises(HTTPException, match="Invalid or expired token") as error:
        auth.verify_token(credentials(token))

    assert error.value.status_code == 401
