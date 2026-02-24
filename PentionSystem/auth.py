import os
import httpx
from jose import jwt, JWTError

KEYCLOAK_URL = os.getenv("KEYCLOAK_URL", "http://localhost:8090")
REALM = os.getenv("KEYCLOAK_REALM")
CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "pention-frontend")

JWKS_URL = f"{KEYCLOAK_URL}/realms/{REALM}/protocol/openid-connect/certs"


async def verify_token(token: str) -> dict:
    async with httpx.AsyncClient() as client:
        jwks = (await client.get(JWKS_URL)).json()
    try:
        payload = jwt.decode(
            token,
            jwks,
            algorithms=["RS256"],
            options={"verify_aud": False}
        )
        return payload
    except JWTError as e:
        raise Exception(f"Token non valido: {e}")


def get_headers(token: str) -> dict:
    """Restituisce gli header con il token da passare agli altri servizi."""
    return {"Authorization": f"Bearer {token}"}