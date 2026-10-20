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
    # JSON list in .env, e.g. cors_origins=["http://localhost:3000"]
    cors_origins: list[str] = ["http://localhost:3000"]

    # Login and password-reset throttling, stored in the broker's Redis.
    rate_limit_enabled: bool = True
    # Email the document owner when processing completes or finally fails.
    processing_email_enabled: bool = True

    # Chat (OpenRouter). An empty key disables chat: POST /api/v1/chat answers 503.
    OPENROUTER_API_KEY: SecretStr = SecretStr("")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    # "openrouter/free" lets OpenRouter pick any free model, so answers vary between
    # calls. Pin an exact model id here before relying on evals.
    CHAT_MODEL: str = "openrouter/free"
    # Fallback providers, tried in this order when the one before them fails. They all
    # speak the OpenAI protocol. A provider without a key is simply skipped.
    GEMINI_AI_API_KEY: SecretStr = SecretStr("")
    GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    CHAT_GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    GROQ_AI_API_KEY: SecretStr = SecretStr("")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    CHAT_GROQ_MODEL: str = "openai/gpt-oss-20b"
    # OPENAI_API_KEY is declared at the top (the document service uses it too). Chat
    # uses it as the last resort, with its own, cheaper model.
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    CHAT_OPENAI_MODEL: str = "gpt-3.5-turbo"
    CHAT_TIMEOUT_SECONDS: float = Field(default=30, gt=0)  # per attempt
    CHAT_TOTAL_TIMEOUT_SECONDS: float = Field(default=45, gt=0)  # all attempts together
    CHAT_MAX_RETRIES: int = Field(default=2, ge=0, le=5)
    # A provider that has a fallback behind it should give up sooner than the last one,
    # and the whole chain gets one time budget.
    CHAT_FALLBACK_MAX_RETRIES: int = Field(default=1, ge=0, le=5)
    CHAT_FALLBACK_TOTAL_TIMEOUT_SECONDS: float = Field(default=90, gt=0)
    # A streamed answer may be long, so it gets its own, larger limit. Silence between
    # two pieces is already capped by CHAT_TIMEOUT_SECONDS.
    CHAT_STREAM_TIMEOUT_SECONDS: float = Field(default=120, gt=0)
    CHAT_MAX_OUTPUT_TOKENS: int = Field(default=1024, gt=0)
    CHAT_RATE_LIMIT_PER_MINUTE: int = Field(default=20, gt=0)

    # Sentry error and performance monitoring. Switched off while SENTRY_DSN is empty.
    SENTRY_DSN: str = ""
    SENTRY_ENVIRONMENT: str = "development"
    SENTRY_RELEASE: str = ""
    SENTRY_TRACES_SAMPLE_RATE: float = Field(default=0.1, ge=0, le=1)
    SENTRY_ENABLE_LOGS: bool = True
    SENTRY_LOGS_LEVEL: str = "INFO"
    SENTRY_DEBUG: bool = False
    # How much of a conversation is sent back to the model: at most this many of the
    # latest messages, and at most this many characters (oldest messages are dropped first).
    CHAT_HISTORY_MESSAGES: int = Field(default=10, ge=0, le=50)
    CHAT_HISTORY_MAX_CHARS: int = Field(default=12000, gt=0)
    # Tokens (input + output) one user may spend per UTC day, across every AI feature. 0 = no limit.
    CHAT_DAILY_TOKEN_BUDGET: int = Field(default=200_000, ge=0)
    # Answers one user may have streaming at the same time (more get a 429).
    CHAT_MAX_CONCURRENT_STREAMS: int = Field(default=2, ge=1)
    TRANSCRIBE_MODEL: str = Field(default="whisper-large-v3-turbo")
    TRANSCRIBE_MAX_FILE_MB: int = Field(default=10, gt=0, le=25)
    TRANSCRIBE_TIMEOUT_SECONDS: int = Field(default=60, gt=0, le=120)
    # "development" shows the values of failed SQL statements in error messages. Anything else,
    # including this default, hides them, because they hold users' messages and transcripts.
    ENVIRONMENT: str = Field(default="production")
    # Image questions use the Gemini key and base URL above. The limit is on the raw file: base64 makes
    # it a third bigger, and Gemini accepts 20 MB for the whole request.
    VISION_MODEL: str = Field(default="gemini-3.1-flash-lite")
    VISION_MAX_IMAGE_MB: int = Field(default=5, gt=0, le=14)
    VISION_TIMEOUT_SECONDS: int = Field(default=60, gt=0, le=120)
    # Virus scanning of uploads with ClamAV (clamd). An empty host switches it off; once a host is set,
    # an unreachable scanner makes uploads fail (503) instead of passing unscanned.
    CLAMAV_HOST: str = ""
    CLAMAV_PORT: int = Field(default=3310, ge=1, le=65535)
    CLAMAV_TIMEOUT_SECONDS: int = Field(default=30, gt=0, le=300)


def load_settings() -> Settings:
    try:
        # Pylance may flag the constructor as needing explicit args even though
        # BaseSettings reads required values from the environment at runtime.
        return Settings()  # type: ignore[call-arg]
    except ValidationError as exc:
        fields = sorted({".".join(str(part)
                        for part in error["loc"]) for error in exc.errors()})
        details = ", ".join(fields)
        raise RuntimeError(
            "Application configuration is invalid. Check the required values in .env; "
            f"missing or invalid settings: {details}"
        ) from None


settings = load_settings()
