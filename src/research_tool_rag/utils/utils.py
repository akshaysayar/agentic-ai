import hashlib
import logging
import re
import string
from functools import partial
from pathlib import Path

strip_whitespace = partial(re.sub, r"\s", "")


def text_hash(text: str, algo=hashlib.md5, strip_all_punctuation=False) -> str:
    assert text
    text = text.strip().lower().replace("'", "")
    normalized = strip_whitespace(text)
    if strip_all_punctuation:
        normalized = "".join(c for c in normalized if c not in string.punctuation)
    return algo(normalized.encode("utf-8")).hexdigest()


def setup_logging(name: str, level: int = logging.INFO, stream_handler: bool = True) -> None:
    """
    Set up logging for the application.

    Args:
        name (str): The name of the logger.
        level (int): The logging level. Defaults to logging.INFO.
    """
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.propagate = False

    path = Path(__file__).parent.parent.parent.parent.resolve() / "logs" / f"{name}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
