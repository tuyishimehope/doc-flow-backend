from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    secret_key: SecretStr
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    app_name: str

    broker_host: str
    broker_backend: str


    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-5.5"
    OPENAI_TIMEOUT_SECONDS: float = Field(default=45, gt=0)
    OPENAI_MAX_INPUT_CHARS: int = Field(default=100_000, gt=0)
    OPENAI_MAX_OUTPUT_TOKENS: int = Field(default=2_048, gt=0)

    DATABASE_NAME: str
    DATABASE_PASSWORD: str
    DATABASE_HOST: str
    DATABASE_PORT: str
    DATABASE_USER: str

    MINIO_ENDPOINT: str
    MINIO_ACCESS_KEY: str
    MINIO_SECRET_KEY: str
    MINIO_BUCKET: str
    MINIO_SECURE: bool

    MAX_UPLOAD_SIZE_BYTES: int = Field(default=10 * 1024 * 1024, gt=0)
    DEPENDENCY_STARTUP_TIMEOUT_SECONDS: int = Field(default=60, gt=0)

    DATABASE_URL_TEST: str

    
    reset_token_expire_minutes: int = 60

    mail_server: str = "localhost"
    mail_port: int = 587
    mail_username: str = ""
    mail_password: SecretStr = SecretStr("")
    mail_from: str = "noreply@example.com"
    mail_use_tls: bool = True

    frontend_url: str = "http://localhost:8000"
    
    
def load_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        details = ", ".join(fields)
        raise RuntimeError(
            "Application configuration is invalid. Check the required values in .env; "
            f"missing or invalid settings: {details}"
        ) from None


settings = load_settings()
