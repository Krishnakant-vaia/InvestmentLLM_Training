import os

_SENTINEL_NAME = "PAUSE"


def _sentinel_path(run_state_dir: str) -> str:
    return os.path.join(run_state_dir, _SENTINEL_NAME)


def pause_requested(run_state_dir: str) -> bool:
    return os.path.exists(_sentinel_path(run_state_dir))


def create_pause_sentinel(run_state_dir: str) -> None:
    os.makedirs(run_state_dir, exist_ok=True)
    open(_sentinel_path(run_state_dir), "a", encoding="utf-8").close()


def clear_pause_sentinel(run_state_dir: str) -> None:
    path = _sentinel_path(run_state_dir)
    if os.path.exists(path):
        os.remove(path)
