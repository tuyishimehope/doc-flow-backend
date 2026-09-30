import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.v1.document.document import router as document_router
from app.api.v1.file.file import router as file_router
from app.api.v1.auth.auth import router as auth_router
from app.api.v1.processing_request.processing_request import router as processing_request_router
from app.db.engine import engine
from app.core.minio import minio_client
from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.wait_for_dependencies import wait_for_dependencies


logger = logging.getLogger(__name__)
configure_logging()


class StartResponse(BaseModel):
    status: str
    service: str
    message: str


@asynccontextmanager    
async def lifespan(app: FastAPI):
    await wait_for_dependencies()
    logger.info("Checking DB connection")

    async with engine.connect() as conn:
        logger.info("Database connected")

    bucket_name = settings.MINIO_BUCKET

    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        logger.info("Created MinIO bucket", extra={"bucket_name": bucket_name})

    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router=auth_router)
app.include_router(router=document_router)
app.include_router(router=file_router)
app.include_router(router=processing_request_router)



@app.get("/", response_model=StartResponse)
def root():
    return start()


@app.get("/start", response_model=StartResponse)
def start():
    return StartResponse(
        status="ok",
        service="doc-flow-backend",
        message="App is running",
    )


@app.get("/health", response_model=StartResponse)
def health():
    return start()
