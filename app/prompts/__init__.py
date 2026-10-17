from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Return the text of ``app/prompts/<name>.md`` (for example ``"chat_v1"``).

    Prompts are read once and cached. A missing or empty file raises, so a bad
    deploy fails loudly instead of sending the model an empty system prompt.
    """
    if Path(name).name != name:
        raise ValueError(f"Invalid prompt name: {name!r}")
    text = (_PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file {name}.md is empty")
    return text
