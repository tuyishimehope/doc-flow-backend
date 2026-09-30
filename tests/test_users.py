import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch

from tests.conftest import auth_header, create_test_user, login_user

api = "api/v1/users"

@pytest.mark.anyio
async def test_signup(client: AsyncClient):
    response = await client.post(f"{api}/signup",
                json={
                    "first_name":"test",
                    "last_name":"test",
                    "email": "test@gmail.com",
                    "password": "12345678"
                })
    assert response.status_code == 201
    
    
@pytest.mark.anyio
async def test_login(client: AsyncClient):
    signup_response = await client.post(f"{api}/signup",
                    json={
                        "first_name":"test",
                        "last_name":"test",
                        "email": "test@gmail.com",
                        "password": "12345678"
                    })
    response = await client.post(f"{api}/token",
                data={
                    "username": "test@gmail.com",
                    "password": "12345678"
                })
    assert response.status_code == 200

@pytest.mark.anyio
async def test_get_current_user(client: AsyncClient, auth_headers: dict[str, str]):
    response = await client.get(f"{api}/me", headers=auth_headers)
    assert response.status_code == 200
    
@pytest.mark.anyio
async def test_signup_failed(client: AsyncClient):
    first_response = await client.post(
        f"{api}/signup",
        json={
            "first_name": "test",
            "last_name": "test",
            "email": "test@gmail.com",
            "password": "12345678",
        },
    )

    response = await client.post(
        f"{api}/signup",
        json={
            "first_name": "test",
            "last_name": "test",
            "email": "test@gmail.com",
            "password": "12345678",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Email already exists"
    
@pytest.mark.anyio
async def test_update_user_info(client: AsyncClient, auth_headers: dict[str, str]):
    
    first_response = await client.get(f"{api}/me", headers=auth_headers)
    data = first_response.json()

    response = await client.patch(f"{api}/{data["id"]}",
                                  json={
                                      "first_name": "user1"
                                      },headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["first_name"] == "user1"