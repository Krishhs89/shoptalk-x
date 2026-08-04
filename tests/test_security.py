"""Guards against a real bug found via live Docker testing: docker-compose's
`${SHOPTALK_API_KEY:-}` substitution sets the container's env var to an
EMPTY STRING (not unset) when the host has no such variable, and
os.environ.get() returns "" (not None) for a set-but-empty var -- a naive
`if API_KEY is None` check let auth silently enforce an empty-string key
that no request could ever match, 401-ing every call in the containerized
deployment."""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _reload_security_with_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SHOPTALK_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SHOPTALK_API_KEY", value)
    if "shoptalk.api.security" in sys.modules:
        del sys.modules["shoptalk.api.security"]
    return importlib.import_module("shoptalk.api.security")


def test_unset_env_var_disables_auth(monkeypatch):
    security = _reload_security_with_env(monkeypatch, None)
    assert security.API_KEY is None
    security.require_api_key(x_api_key=None)  # must not raise


def test_empty_string_env_var_disables_auth(monkeypatch):
    """The docker-compose ${VAR:-} case -- set but empty."""
    security = _reload_security_with_env(monkeypatch, "")
    assert security.API_KEY is None
    security.require_api_key(x_api_key=None)  # must not raise


def test_real_key_enforces_auth(monkeypatch):
    security = _reload_security_with_env(monkeypatch, "secret123")
    assert security.API_KEY == "secret123"
    security.require_api_key(x_api_key="secret123")  # must not raise

    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        security.require_api_key(x_api_key="wrong")
    assert exc_info.value.status_code == 401
