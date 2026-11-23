"""Service and admin bearer tokens (CRD 10 / LLD 4.1; production: mTLS + 15-min JWT, SSO + MFA for admins).

The profile's `api.token` / `api.admin_token` are demo values committed to git. On any host other people can reach
(a GPU VM, AWS), ARGUS_SCORING_TOKEN / ARGUS_SCORING_ADMIN_TOKEN (.env) replace them; the platform API reads the
same two variables, so both sides always agree.
"""
from __future__ import annotations

import hmac
import os

TOKEN_ENV = {"token": "ARGUS_SCORING_TOKEN", "admin_token": "ARGUS_SCORING_ADMIN_TOKEN"}


def apply_token_env(api) -> None:
    """Tokens from the environment win over the profile's (in place)."""
    for key, env in TOKEN_ENV.items():
        if os.environ.get(env):
            api[key] = os.environ[env]


def bearer_ok(authorization: str | None, token: str) -> bool:
    """Constant-time check of an `Authorization: Bearer <token>` header."""
    return hmac.compare_digest((authorization or "").encode(), f"Bearer {token}".encode())


def required_token(path: str, api) -> str | None:
    """The token a path needs: the admin token for /admin/, the service token for /v1/fraud/, none otherwise
    (health, OpenAPI docs)."""
    if path.startswith("/admin/"):
        return api["admin_token"]
    if path.startswith("/v1/fraud/"):
        return api["token"]
    return None


def authorised(path: str, authorization: str | None, api) -> bool:
    """Checked before routing and body validation, so a caller without the token learns nothing (401, never 400)."""
    token = required_token(path, api)
    return token is None or bearer_ok(authorization, token)
