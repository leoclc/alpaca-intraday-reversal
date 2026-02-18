import json

from app.config import loader


def test_load_config_accepts_app_config_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "candidate.json"
    cfg_path.write_text(json.dumps({"__probe_config_file__": "ok"}), encoding="utf-8")

    monkeypatch.delenv("APP_CONFIG_PATH", raising=False)
    monkeypatch.delenv("ALPACA_OHLC_CONFIG", raising=False)
    monkeypatch.setenv("APP_CONFIG_FILE", str(cfg_path))
    loader._LOGGED_PATHS = False

    cfg = loader.load_config()
    assert cfg.get("__probe_config_file__") == "ok"
