
from pathlib import Path


def get_test_root() -> Path:
    root_path = Path(__file__).parent

    return root_path
