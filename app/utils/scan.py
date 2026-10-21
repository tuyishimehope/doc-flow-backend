"""Virus scan of uploaded bytes with ClamAV's ``clamd`` daemon.

One place for every feature that takes a file.

Scanning is on when ``CLAMAV_HOST`` is set and off when it is empty (the same convention as
``SENTRY_DSN`` and the provider API keys). Once it is on it fails closed: if the scanner cannot
be reached, or answers with anything but a clean verdict, the upload is refused with a 503
instead of being let through.

Call ``scan_upload(content)`` after the cheap checks of ``app/utils/upload.py`` and *before*
anything parses the file (an image decoder, a PDF reader), because parsers are what a hostile
file attacks.

The wire protocol is clamd's ``INSTREAM`` command (docs.clamav.net, "ClamD protocol"): send
``zINSTREAM\\0``, then chunks of a 4-byte big-endian length plus data, then a zero length. The
reply is ``stream: OK`` or ``stream: <signature> FOUND``, ended by a NUL byte. The file must
stay under ``StreamMaxLength`` in ``clamd.conf``; keep the upload limits below it. Nothing here
logs file content or file names.
"""

import asyncio
import logging
import struct
import time

from app.core.config import settings
from app.utils.upload import UploadRejectedError

logger = logging.getLogger(__name__)

CHUNK_BYTES = 64 * 1024
PROVIDER = "clamav"
_warned_off = False


async def scan_upload(content: bytes) -> None:
    """Return normally for a clean file.

    Raise ``UploadRejectedError`` for a virus (422) or a scanner problem (503).
    """
    host = settings.CLAMAV_HOST.strip()
    if not host:
        _warn_scanning_is_off_once()
        return

    started = time.monotonic()
    try:
        reply = await asyncio.wait_for(
            _instream(host, settings.CLAMAV_PORT, content),
            timeout=settings.CLAMAV_TIMEOUT_SECONDS,
        )
    except (OSError, EOFError, asyncio.TimeoutError, asyncio.LimitOverrunError):
        _log(logging.ERROR, "Virus scanner unreachable", "unavailable", started, exc_info=True)
        raise UploadRejectedError(
            503, "File scanning is unavailable right now. Please try again shortly."
        ) from None

    verdict = reply.rstrip(b"\0\n").decode("utf-8", "replace").removeprefix("stream: ")
    if verdict == "OK":
        _log(logging.INFO, "Upload scanned", "ok", started)
        return
    if verdict.endswith(" FOUND"):
        # The signature name is ClamAV's, not the user's data, so it is safe to log.
        _log(
            logging.WARNING,
            f"Upload rejected by virus scan: {verdict.removesuffix(' FOUND')}",
            "infected",
            started,
        )
        raise UploadRejectedError(422, "The file was rejected by the virus scan.")
    _log(
        logging.ERROR,
        f"Virus scanner gave an unexpected answer: {verdict[:100]}",
        "unavailable",
        started,
    )
    raise UploadRejectedError(
        503, "File scanning is unavailable right now. Please try again shortly."
    )


async def _instream(host: str, port: int, content: bytes) -> bytes:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(b"zINSTREAM\0")
        for start in range(0, len(content), CHUNK_BYTES):
            chunk = content[start : start + CHUNK_BYTES]
            writer.write(struct.pack("!I", len(chunk)) + chunk)
            await writer.drain()
        writer.write(struct.pack("!I", 0))
        await writer.drain()
        return await reader.readuntil(b"\0")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass


def _log(level: int, message: str, status: str, started: float, **kwargs) -> None:
    logger.log(
        level,
        message,
        extra={
            "provider": PROVIDER,
            "status": status,
            "latency_ms": int((time.monotonic() - started) * 1000),
        },
        **kwargs,
    )


def _warn_scanning_is_off_once() -> None:
    global _warned_off
    if not _warned_off:
        _warned_off = True
        logger.warning(
            "Virus scanning is off because CLAMAV_HOST is empty; uploads are not scanned"
        )
