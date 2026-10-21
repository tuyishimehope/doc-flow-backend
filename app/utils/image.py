"""Image uploads: what is accepted, and how an accepted image is made safe to pass on.

``image_policy`` plugs into ``read_checked_upload`` (``app/utils/upload.py``). ``clean_image`` then
decodes the picture and writes it again, as the OWASP cheat sheet advises:

* All metadata is dropped: EXIF (GPS position, camera, time), text chunks, thumbnails.
* Anything that is not pixel data, such as bytes appended after the image, is dropped.
* The photo is turned upright first, so removing the orientation tag does not rotate it.
* A picture with too many pixels is refused before it is decoded (a small file can unpack to
  gigabytes).

Re-encoding is not a guarantee that nothing harmful survives; it removes the common tricks.
"""

import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from app.core.config import settings
from app.utils.chat_utils import convert_mb_to_bytes
from app.utils.upload import FileType, UploadPolicy, UploadRejectedError

MAX_IMAGE_PIXELS = 40_000_000  # about 6,300 x 6,300; a 48 megapixel phone photo is refused

_JPEG = FileType("image/jpeg", lambda b: b.startswith(b"\xff\xd8\xff"))
_PNG = FileType("image/png", lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"))
_WEBP = FileType("image/webp", lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP")

IMAGE_TYPES = {"jpg": _JPEG, "jpeg": _JPEG, "png": _PNG, "webp": _WEBP}
_MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def image_policy() -> UploadPolicy:
    return UploadPolicy(
        types=IMAGE_TYPES, max_bytes=convert_mb_to_bytes(settings.VISION_MAX_IMAGE_MB)
    )


@dataclass(frozen=True)
class CleanImage:
    content: bytes
    mime: str
    width: int
    height: int


def clean_image(content: bytes) -> CleanImage:
    """Decode and re-encode.

    Raises ``UploadRejectedError`` for anything that is not a usable image.
    """
    try:
        with Image.open(io.BytesIO(content)) as opened:
            image_format = opened.format
            if image_format not in _MIME_BY_FORMAT:
                raise UploadRejectedError(415, "This image format is not supported.")
            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise UploadRejectedError(
                    413, "The picture has too many pixels. Use a smaller photo."
                )
            image = ImageOps.exif_transpose(opened)  # decodes; a copy, so the original can close
            if image_format == "JPEG" and image.mode not in ("RGB", "L"):
                image = image.convert(
                    "RGB"
                )  # JPEG cannot hold transparency or CMYK from other formats
            out = io.BytesIO()
            if image_format == "PNG":
                image.save(out, "PNG")
            else:
                image.save(out, image_format, quality=90)
            return CleanImage(
                out.getvalue(), _MIME_BY_FORMAT[image_format], image.width, image.height
            )
    except UploadRejectedError:
        raise
    except Exception:  # Pillow raises many kinds of errors for broken or hostile files
        raise UploadRejectedError(422, "The file is not a valid image.") from None
