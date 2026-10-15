"""Authentication and authorization dependency for ClaimAssist AI."""

from typing import Optional, Dict
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

security = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    """Represents a verified, authenticated principal."""
    user_id: str
    policyholder_id: Optional[str] = None
    policyholder_name: Optional[str] = None
    role: str = "member"  # "member", "staff", "admin"

    @property
    def is_admin(self) -> bool:
        return self.role in ("admin", "staff")


# Known credentials registry for demonstration and production token resolution
# Tokens map to verified identities; in enterprise deployments this verifies JWT signatures / OIDC
KNOWN_TOKENS: Dict[str, AuthenticatedUser] = {
    # Admin / Staff token
    "admin-secret-token": AuthenticatedUser(
        user_id="USR-ADMIN-1",
        role="admin",
        policyholder_name="Specialist Miller"
    ),
    "staff-token-123": AuthenticatedUser(
        user_id="USR-STAFF-1",
        role="staff",
        policyholder_name="Senior Adjudicator"
    ),
    # Member tokens mapped to specific policyholders
    "token-pol-98214": AuthenticatedUser(
        user_id="USR-98214",
        policyholder_id="POL-98214",
        policyholder_name="Jordan Miller",
        role="member"
    ),
    "token-pol-77312": AuthenticatedUser(
        user_id="USR-77312",
        policyholder_id="POL-77312",
        policyholder_name="Samantha Chen",
        role="member"
    ),
    "token-pol-45109": AuthenticatedUser(
        user_id="USR-45109",
        policyholder_id="POL-45109",
        policyholder_name="Marcus Vance",
        role="member"
    ),
    "token-pol-61028": AuthenticatedUser(
        user_id="USR-61028",
        policyholder_id="POL-61028",
        policyholder_name="Elena Rostova",
        role="member"
    ),
    "token-pol-88391": AuthenticatedUser(
        user_id="USR-88391",
        policyholder_id="POL-88391",
        policyholder_name="David Kim",
        role="member"
    ),
    "token-pol-33290": AuthenticatedUser(
        user_id="USR-33290",
        policyholder_id="POL-33290",
        policyholder_name="Alicia Gomez",
        role="member"
    ),
    "token-pol-88123": AuthenticatedUser(
        user_id="USR-88123",
        policyholder_id="POL-88123",
        policyholder_name="Elena Gomez",
        role="member"
    ),
    "token-pol-33291": AuthenticatedUser(
        user_id="USR-33291",
        policyholder_id="POL-33291",
        policyholder_name="David Kim",
        role="member"
    ),
    "token-pol-61044": AuthenticatedUser(
        user_id="USR-61044",
        policyholder_id="POL-61044",
        policyholder_name="Priya Patel",
        role="member"
    ),
}


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> AuthenticatedUser:
    """
    Validates Bearer token credentials and extracts the authenticated principal.
    Raises HTTP 401 if credentials are missing or invalid.
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Missing or empty Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials.strip()

    # 1. Exact match in known tokens
    if token in KNOWN_TOKENS:
        return KNOWN_TOKENS[token]

    # 2. Support standard prefix convention 'member-POL-XXXXX' or raw policy ID as token
    token_clean = token.replace("member-", "").strip().upper()
    for auth_user in KNOWN_TOKENS.values():
        if auth_user.policyholder_id and auth_user.policyholder_id.upper() == token_clean:
            return auth_user

    # 3. Invalid credentials
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials or expired Bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )
