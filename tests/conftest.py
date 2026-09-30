import os
from collections.abc import AsyncGenerator
from unittest.mock import patch

# Set test credentials before importing app settings, while preserving the
# caller's DATABASE_URL_TEST so the test database is explicitly configured.
os.environ["secret_key"] = "test-secret-key-for-testing-only"
os.environ["MINIO_ENDPOINT"] = "localhost:9000"
os.environ["MINIO_ACCESS_KEY"] = "minioadmin"
os.environ["MINIO_SECRET_KEY"] = "minioadmin"
os.environ["MINIO_BUCKET"] = "test-bucket"
os.environ["MINIO_SECURE"] = "false"

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from httpx import ASGITransport, AsyncClient
import pytest

from app.core.config import settings
from app.main import app
from app.db.dependencies import get_db_session
from app.db.base_class import Base


TEST_DATABASE_URL = settings.DATABASE_URL_TEST
test_database_name = make_url(TEST_DATABASE_URL).database or ""
test_database_name_is_marked = (
    test_database_name == "test"
    or test_database_name.startswith("test_")
    or test_database_name.endswith("_test")
)
if not test_database_name or (
    not test_database_name_is_marked
    and test_database_name == settings.DATABASE_NAME
):
    raise RuntimeError(
        "Refusing to reset the application database. Set DATABASE_URL_TEST "
        "to a separate database dedicated to tests."
    )


pytest_plugins = ["anyio"]


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"


@pytest.fixture(scope="session")
def test_engine():
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
    )
    return engine


@pytest.fixture(scope="session")
async def setup_database(test_engine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

    yield

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

    await test_engine.dispose()


@pytest.fixture
async def db_session(
    test_engine,
    setup_database,
) -> AsyncGenerator[AsyncSession]:
    conn = await test_engine.connect()
    trans = await conn.begin()

    test_async_session = async_sessionmaker(
        bind=conn,
        class_=AsyncSession,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )

    async with test_async_session() as session:
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
            await conn.close()


@pytest.fixture
def mocked_minio():
    with patch("app.service.file.file.minio_client") as mock_client:
        yield mock_client


@pytest.fixture
async def client(
    db_session: AsyncSession,
    mocked_minio,
) -> AsyncGenerator[AsyncClient]:

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = override_get_db

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


async def create_test_user(
    client: AsyncClient,
    first_name: str = "testuser",
    last_name: str = "testuser",
    email: str = "test@example.com",
    password: str = "testpassword123",
) -> dict:
    response = await client.post(
        "/api/v1/users/signup",
        json={
            "first_name": first_name,
            "last_name": last_name,
            "email": email,
            "password": password,
        },
    )
    assert response.status_code == 201, f"Failed to create user: {response.text}"
    return response.json()


async def login_user(
    client: AsyncClient,
    email: str = "test@example.com",
    password: str = "testpassword123",
) -> str:
    response = await client.post(
        "/api/v1/users/token",
        data={
            "username": email,
            "password": password,
        },
    )
    assert response.status_code == 200, f"Failed to login: {response.text}"
    return response.json()["access_token"]


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def auth_headers(client: AsyncClient) -> dict[str, str]:
    await create_test_user(client)

    token = await login_user(client)

    return auth_header(token)
