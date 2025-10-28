import os
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

_LOCK_PATH = Path(os.environ.get("LLAMA_CPP_LOCK_PATH", ".llamacpp.lock"))
_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock = FileLock(str(_LOCK_PATH))

@contextmanager
def llama_cpp_lock():
    with _lock:
        yield
