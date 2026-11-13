import os
import sys
import types

from meridian import cli


def test_serve_passes_config_to_server(monkeypatch, tmp_path):
    cfg = tmp_path / "m.toml"
    cfg.write_text("")
    seen = {}
    def run(*a, **k):
        seen["env"] = os.environ.get("MERIDIAN_CONFIG")

    fake = types.SimpleNamespace(run=run)
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    monkeypatch.delenv("MERIDIAN_CONFIG", raising=False)
    assert cli.main(["--config", str(cfg), "serve", "--port", "0"]) == 0
    assert seen["env"] == str(cfg.resolve())
