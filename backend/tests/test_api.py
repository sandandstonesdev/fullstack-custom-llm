"""API smoke test: startup, health probe and the WebSocket message contract."""
from fastapi.testclient import TestClient

from app.main import app


def test_api_endpoints():
    with TestClient(app) as client:
        # Startup: model may or may not be trained yet - either way the API serves.
        assert client.get("/").status_code == 200

        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert isinstance(body["model_loaded"], bool)

        # Bad payload -> structured error reply, connection stays usable.
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not json")
            reply = ws.receive_json()
            assert "error" in reply
            assert reply["isUser"] is False

        # Valid payload: bot reply when a model is loaded
        # (otherwise the server is in silent relay mode for other clients).
        if body["model_loaded"]:
            with client.websocket_connect("/ws") as ws:
                ws.send_json({"text": "Polska to", "isUser": True})
                reply = ws.receive_json()
                assert reply["isUser"] is False
                assert "text" in reply
