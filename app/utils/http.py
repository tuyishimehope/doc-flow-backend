from urllib.parse import quote


def attachment_disposition(filename: str) -> str:
    """Content-Disposition value that survives quotes and non-ASCII filenames."""
    fallback = "".join(
        char for char in filename if char.isascii() and char.isprintable() and char not in '"\\'
    ) or "download"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"
