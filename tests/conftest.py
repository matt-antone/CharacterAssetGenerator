import pytest

from cag.publish import REMOTE_ENV
from cag.pull import REMOTE_ENV as MOTION_REMOTE_ENV


@pytest.fixture(autouse=True)
def no_output_remote(monkeypatch):
    """A test never publishes, or pulls, because the shell running it has a remote set."""
    monkeypatch.delenv(REMOTE_ENV, raising=False)
    monkeypatch.delenv(MOTION_REMOTE_ENV, raising=False)
