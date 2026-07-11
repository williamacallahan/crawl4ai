import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import EmailStr
from pydantic.main import BaseModel

security = HTTPBearer(auto_error=False)
ACCESS_TOKEN_EXPIRE_MINUTES = 60

_WEAK_SECRETS = {"mysecret", "secret", "password", "changeme", "test", "12345678"}


def _resolve_secret_key() -> str:
    """Resolve SECRET_KEY: validate if set, auto-generate if JWT enabled but unset."""
    import logging
    import secrets as _secrets
    key = os.environ.get("SECRET_KEY", "")
    if key:
        if key.lower() in _WEAK_SECRETS:
            raise RuntimeError(
                "FATAL: SECRET_KEY is a known weak value. "
                "Generate a strong one: python3 -c \"import secrets; print(secrets.token_hex(32))\""
            )
        if len(key) < 32:
            raise RuntimeError(
                "FATAL: SECRET_KEY must be at least 32 characters. "
                "Generate one: python3 -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return key
    # No key set -- auto-generate ephemeral key
    generated = _secrets.token_hex(32)
    logging.getLogger("crawl4ai.security").warning(
        "No SECRET_KEY set. Auto-generated ephemeral key (changes on restart). "
        "Set SECRET_KEY env var for production."
    )
    return generated


SECRET_KEY = _resolve_secret_key()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with an expiration."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")

def verify_token(credentials: HTTPAuthorizationCredentials) -> Dict:
    """Verify the JWT token from the Authorization header."""
    
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=401, 
            detail="No token provided",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.PyJWTError as e:
        raise HTTPException(
            status_code=401, 
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


def get_token_dependency(config: Dict):
    """Return the token dependency if JWT is enabled, else a function that returns None."""
    
    if config.get("security", {}).get("jwt_enabled", False):
        def jwt_required(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
            """Enforce JWT authentication when enabled."""
            if credentials is None:
                raise HTTPException(
                    status_code=401, 
                    detail="Authentication required. Please provide a valid Bearer token.",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            return verify_token(credentials)
        return jwt_required
    else:
        return lambda: None


class TokenRequest(BaseModel):
    email: EmailStr
    api_token: Optional[str] = None
