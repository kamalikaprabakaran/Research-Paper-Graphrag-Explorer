import os
from pathlib import Path
from config.settings import Settings


def test_settings_load_without_credentials(monkeypatch):
    # Ensure no environment variables are set
    for var in ["OPENROUTER_API_KEY", "OPENROUTER_MODEL", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]:
        monkeypatch.delenv(var, raising=False)
    settings = Settings()
    assert settings.OPENROUTER_API_KEY is None
    assert settings.NEO4J_URI is None


def test_env_file_loading(tmp_path, monkeypatch):
    env_content = """
OPENROUTER_API_KEY=demo_key
OPENROUTER_MODEL=gpt-4o-mini
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=secret
"""
    env_path = tmp_path / ".env"
    env_path.write_text(env_content)
    monkeypatch.chdir(tmp_path)
    settings = Settings(_env_file=str(env_path))
    assert settings.OPENROUTER_API_KEY == "demo_key"
    assert settings.NEO4J_URI == "bolt://localhost:7687"
