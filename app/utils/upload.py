"""Checks for files that users upload. One place for every feature that takes a file.

Based on the OWASP File Upload Cheat Sheet. What this module does, in this order:

1. The file name is only a label. Its last extension must be on the policy's allowlist (415).
2. The size is checked while reading, never trusted from the client (413), and an empty file is refused (422).
3. The first bytes must match the type the extension claims (415). The browser's content type is ignored.
4. A SHA-256 of the bytes is kept for audit and duplicate detection.

What it does not do (add it where the trigger is true): scan for viruses, strip active content from PDF or
Office files, or limit the size of the whole request. Set that last limit in the reverse proxy
(for example ``client_max_body_size`` in nginx), because a server has read the whole upload by the time this runs.

A feature describes what it accepts with an ``UploadPolicy`` and gets back a ``CheckedUpload``.
Images live in ``app/utils/image.py``; a PDF, DOCX or text policy for document features goes the same way.
"""

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePath

from fastapi import UploadFile

MAX_DISPLAY_NAME_CHARS = 100


class UploadRejected(Exception):
    """A file that must not be used. ``status_code`` is the HTTP status it deserves."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class FileType:
    mime: str
    matches: Callable[[bytes], bool]  # looks at the first bytes only


@dataclass(frozen=True)
class UploadPolicy:
    types: Mapping[str, FileType]  # extension (lower case, no dot) -> what that extension must contain
    max_bytes: int


@dataclass(frozen=True)
class CheckedUpload:
    content: bytes
    extension: str
    mime: str  # from the type table, never from the client
    size: int
    sha256: str
    display_name: str  # a label for people; never use it as a path


async def read_checked_upload(file: UploadFile, policy: UploadPolicy) -> CheckedUpload:
    name = file.filename or ""
    if not name or "\x00" in name:
        raise UploadRejected(422, "Please upload a file.")
    extension = PurePath(name.replace("\\", "/")).suffix.lstrip(".").lower()
    file_type = policy.types.get(extension)
    if file_type is None:
        allowed = ", ".join(sorted(policy.types))
        raise UploadRejected(415, f"This file type is not supported. Use one of: {allowed}.")

    # One byte more than the limit is enough to know it is too big, so memory stays bounded.
    content = await file.read(policy.max_bytes + 1)
    if len(content) > policy.max_bytes:
        raise UploadRejected(413, f"The file is too big. The limit is {_megabytes(policy.max_bytes)} MB.")
    if not content:
        raise UploadRejected(422, "The file is empty.")
    if not file_type.matches(content):
        raise UploadRejected(415, f"The content of the file is not a real .{extension} file.")

    return CheckedUpload(
        content=content,
        extension=extension,
        mime=file_type.mime,
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        display_name=_display_name(name),
    )


def _display_name(name: str) -> str:
    base = PurePath(name.replace("\\", "/")).name
    return re.sub(r"[\x00-\x1f\x7f]", "", base)[:MAX_DISPLAY_NAME_CHARS]


def _megabytes(size: int) -> str:
    return f"{size / (1024 * 1024):g}"
