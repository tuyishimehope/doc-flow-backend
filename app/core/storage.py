"""The S3 client for object storage (MinIO locally, any S3-compatible service elsewhere)."""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.core.config import settings

# Codes S3 and MinIO answer with when a bucket or object does not exist.
MISSING_CODES = {"404", "NoSuchKey", "NoSuchBucket", "NotFound"}


def endpoint_url(endpoint: str, secure: bool) -> str:
    """A full URL for boto3: a bare host:port gets http/https from MINIO_SECURE."""
    if "://" in endpoint:
        return endpoint.rstrip("/")
    return f"{'https' if secure else 'http'}://{endpoint}"


s3_client = boto3.client(
    "s3",
    endpoint_url=endpoint_url(settings.MINIO_ENDPOINT, settings.MINIO_SECURE),
    aws_access_key_id=settings.MINIO_ACCESS_KEY,
    aws_secret_access_key=settings.MINIO_SECRET_KEY,
    region_name=settings.MINIO_REGION,
    config=Config(
        signature_version="s3v4",
        # MinIO serves buckets as a path (host/bucket/key), not as a subdomain.
        s3={"addressing_style": "path"},
        connect_timeout=5,
        read_timeout=60,
        retries={"max_attempts": 3, "mode": "standard"},
    ),
)


def is_missing(error: ClientError) -> bool:
    """Whether the error says the bucket or object does not exist."""
    return str(error.response.get("Error", {}).get("Code")) in MISSING_CODES


def bucket_exists(bucket: str) -> bool:
    try:
        s3_client.head_bucket(Bucket=bucket)
    except ClientError as error:
        if is_missing(error):
            return False
        raise
    return True


def ensure_bucket(bucket: str) -> bool:
    """Create the bucket if it is not there. True when it was created."""
    if bucket_exists(bucket):
        return False
    s3_client.create_bucket(Bucket=bucket)
    return True
