from email.message import EmailMessage

import aiosmtplib
from fastapi.templating import Jinja2Templates

from app.core.config import settings

templates = Jinja2Templates(directory="templates")


async def send_email(
    to_email: str,
    subject: str,
    plain_text: str,
    html_content: str | None = None,
) -> None:
    message = EmailMessage()
    message["From"] = settings.mail_from
    message["To"] = to_email
    message["Subject"] = subject

    message.set_content(plain_text)

    if html_content:
        message.add_alternative(html_content, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.mail_server,
        port=settings.mail_port,
        username=settings.mail_username or None,
        password=settings.mail_password.get_secret_value() or None,
        start_tls=settings.mail_use_tls,
        timeout=10,
    )


async def send_password_reset_email(to_email: str, username: str, token: str) -> None:
    reset_url = f"{settings.frontend_url}/reset-password?token={token}"

    template = templates.env.get_template("email/password_reset.html")
    expire_minutes = settings.reset_token_expire_minutes
    html_content = template.render(
        reset_url=reset_url,
        username=username,
        app_name=settings.app_name,
        expire_minutes=expire_minutes,
    )

    plain_text = f"""Hi {username},

You requested to reset your password. Click the link below to set a new password:

{reset_url}

This link will expire in {expire_minutes} minutes.

If you didn't request this, you can safely ignore this email.

Best regards,
The {settings.app_name} Team
"""

    await send_email(
        to_email=to_email,
        subject=f"Reset Your Password - {settings.app_name}",
        plain_text=plain_text,
        html_content=html_content,
    )


async def send_processing_finished_email(
    to_email: str,
    first_name: str,
    document_name: str,
    processing_type: str,
    succeeded: bool,
    failure_reason: str | None = None,
) -> None:
    task = processing_type.replace("_", " ").lower()
    if succeeded:
        subject = f"Your document is ready - {settings.app_name}"
        outcome = f"We finished the {task} for \"{document_name}\"."
    else:
        subject = f"We could not process your document - {settings.app_name}"
        outcome = f"The {task} for \"{document_name}\" failed: {failure_reason}"

    plain_text = f"""Hi {first_name},

{outcome}

Open {settings.app_name}: {settings.frontend_url}

Best regards,
The {settings.app_name} Team
"""
    await send_email(to_email=to_email, subject=subject, plain_text=plain_text)
