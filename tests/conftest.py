import pytest

from cag.publish import REMOTE_ENV


@pytest.fixture(autouse=True)
def no_output_remote(monkeypatch):
    """A test never publishes because the shell running it has a remote set."""
    monkeypatch.delenv(REMOTE_ENV, raising=False)
