"""Single shared tiktoken GPT-2 encoding.

Pins TIKTOKEN_CACHE_DIR to a project-local directory so the BPE files are
downloaded once and survive TEMP being cleared; a half-written cache or a
dropped connection would otherwise fail every dataset prepare run.
"""

import os
import time

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tiktoken_cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", CACHE_DIR)

import tiktoken  # noqa: E402  (import after the cache dir is set)

ENCODING_NAME = "gpt2"
_encoding = None


def get_encoding(retries: int = 5, backoff: float = 3.0):
    """Return the shared GPT-2 encoding, retrying transient network errors."""
    global _encoding
    if _encoding is not None:
        return _encoding

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            _encoding = tiktoken.get_encoding(ENCODING_NAME)
            return _encoding
        except Exception as e:  # network / partial-cache failures
            last_error = e
            if attempt < retries:
                time.sleep(backoff)
    raise RuntimeError(
        f"Could not load the '{ENCODING_NAME}' tokenizer after {retries} attempts: {last_error}"
    )
