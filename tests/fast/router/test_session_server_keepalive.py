"""The session server must hold idle connections longer than its HTTP clients reuse them."""

from types import SimpleNamespace

import httpx

from miles.rollout.session import server


def test_server_keeps_idle_connections_longer_than_httpx_clients(monkeypatch):
    served = {}
    monkeypatch.setattr(server, "SessionServer", lambda config: SimpleNamespace(app="session-app"))
    monkeypatch.setattr(server.uvicorn, "run", lambda app, **kwargs: served.update(app=app, **kwargs))
    monkeypatch.setattr(server.setproctitle, "setproctitle", lambda title: None)

    server.run_session_server(SimpleNamespace(host="127.0.0.1", port=0, backend_url="http://router"))

    assert served["app"] == "session-app"
    # Otherwise a turn that starts as the server closes an idle socket fails with httpx.ReadError.
    assert served["timeout_keep_alive"] > httpx.Limits().keepalive_expiry
