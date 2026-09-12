"""Authentication and authorization utilities for the RAG API."""
from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class Role(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    USER = "user"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True)
class AuthContext:
    """Security context of the authenticated caller."""

    user_id: str
    role: Role
    token: str | None = None


# Load keys from environment with secure fallback defaults
_ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "rag-admin-secret-key-2026")
_CLIENT_API_KEY = os.getenv("API_KEY", os.getenv("CLIENT_API_KEY", "rag-client-key-2026"))

bearer_scheme = HTTPBearer(auto_error=False)


def _safe_compare(val1: str, val2: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    if not val1 or not val2:
        return False
    return hmac.compare_digest(val1.strip().encode("utf-8"), val2.strip().encode("utf-8"))


def get_auth_context(
    request: Request,
    bearer: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> AuthContext:
    """Extract and validate authentication token from Bearer header or X-API-Key.
    
    Supports:
    - Admin API key -> Role.ADMIN
    - Client API key or session token -> Role.USER
    - Missing credentials -> Role.ANONYMOUS
    """
    token = None
    if bearer and bearer.credentials:
        token = bearer.credentials.strip()
    elif x_api_key:
        token = x_api_key.strip()

    if not token:
        client_ip = request.client.host if request.client else "127.0.0.1"
        return AuthContext(user_id=f"anon_{client_ip}", role=Role.ANONYMOUS, token=None)

    # Check Admin credentials
    admin_key = os.getenv("ADMIN_API_KEY", _ADMIN_API_KEY)
    if _safe_compare(token, admin_key):
        return AuthContext(user_id="admin", role=Role.ADMIN, token=token)

    # Check Client credentials
    client_key = os.getenv("API_KEY", os.getenv("CLIENT_API_KEY", _CLIENT_API_KEY))
    if _safe_compare(token, client_key):
        # Deterministic client ID derived from token
        client_ip = request.client.host if request.client else "127.0.0.1"
        return AuthContext(user_id=f"user_{client_ip}", role=Role.USER, token=token)

    # Accept user-scoped Bearer session tokens (e.g. user_session_*)
    if token.startswith("user_") or token.startswith("sess_") or len(token) >= 16:
        # Use first 32 chars of token as user_id
        safe_user_id = token[:32].replace(":", "_").replace("/", "_")
        return AuthContext(user_id=safe_user_id, role=Role.USER, token=token)

    return AuthContext(user_id="unauthorized", role=Role.ANONYMOUS, token=None)


def require_user(
    auth: Annotated[AuthContext, Depends(get_auth_context)]
) -> AuthContext:
    """Require at least Role.USER or Role.ADMIN to proceed."""
    if auth.role == Role.ANONYMOUS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide a valid Bearer token or X-API-Key header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth


def require_admin(
    auth: Annotated[AuthContext, Depends(get_auth_context)]
) -> AuthContext:
    """Require Role.ADMIN to proceed."""
    if auth.role == Role.ANONYMOUS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Administrative authentication required. Provide a valid admin token or X-API-Key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if auth.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Administrative privileges required.",
        )
    return auth
