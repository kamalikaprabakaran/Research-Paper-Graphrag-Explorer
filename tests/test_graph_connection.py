# tests/test_graph_connection.py
"""Unit tests for the Neo4jManager connection manager.
These tests use mocks to avoid any real network calls or credential requirements.
"""

import builtins
from unittest import mock
import pytest

from graph import Neo4jManager, Neo4jConfigurationError

# Helper to create a dummy Settings object with custom values
class DummySettings:
    def __init__(self, uri=None, username=None, password=None):
        self.NEO4J_URI = uri
        self.NEO4J_USERNAME = username
        self.NEO4J_PASSWORD = password

def test_construction_does_not_create_driver(monkeypatch):
    # Patch GraphDatabase.driver to raise if called
    with mock.patch("neo4j.GraphDatabase.driver") as mock_driver:
        manager = Neo4jManager(settings=DummySettings("bolt://localhost:7687", "user", "pwd"))
        # No driver should be created yet
        mock_driver.assert_not_called()
        # Accessing driver property should create it
        _ = manager.driver
        mock_driver.assert_called_once()

def test_injected_driver_is_reused(monkeypatch):
    mock_driver = mock.MagicMock()
    manager = Neo4jManager(settings=DummySettings("uri", "u", "p"), driver=mock_driver, own_driver=False)
    # Driver should be the same object
    assert manager.driver is mock_driver
    # GraphDatabase.driver should never be called
    with mock.patch("neo4j.GraphDatabase.driver") as driver_factory:
        assert manager.driver is mock_driver
        driver_factory.assert_not_called()

@pytest.mark.parametrize(
    "uri,username,password,missing",
    [
        (None, "u", "p", ["NEO4J_URI"]),
        ("uri", None, "p", ["NEO4J_USERNAME"]),
        ("uri", "u", None, ["NEO4J_PASSWORD"]),
        ("", "u", "p", ["NEO4J_URI"]),
        ("uri", "", "p", ["NEO4J_USERNAME"]),
        ("uri", "u", "", ["NEO4J_PASSWORD"]),
    ],
)
def test_missing_credentials_raises(uri, username, password, missing):
    settings = DummySettings(uri, username, password)
    manager = Neo4jManager(settings=settings)
    with pytest.raises(Neo4jConfigurationError) as exc:
        _ = manager.driver
    assert set(exc.value.missing) == set(missing)

def test_lazy_creation_once(monkeypatch):
    with mock.patch("neo4j.GraphDatabase.driver") as driver_factory:
        driver_instance = mock.MagicMock()
        driver_factory.return_value = driver_instance
        manager = Neo4jManager(settings=DummySettings("bolt://localhost", "u", "p"))
        # First access creates driver
        d1 = manager.driver
        # Second access returns same instance, no extra call
        d2 = manager.driver
        assert d1 is d2
        driver_factory.assert_called_once()

def test_verify_connection_success(monkeypatch):
    mock_session = mock.MagicMock()
    mock_session.run.return_value.single.return_value = mock.MagicMock()
    mock_driver = mock.MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    manager = Neo4jManager(settings=DummySettings("uri", "u", "p"), driver=mock_driver, own_driver=False)
    assert manager.verify_connection() is True
    mock_driver.session.assert_called_once()
    mock_session.run.assert_called_once_with("RETURN 1")

def test_verify_connection_failure(monkeypatch):
    mock_driver = mock.MagicMock()
    mock_driver.session.side_effect = Exception("boom")
    manager = Neo4jManager(settings=DummySettings("uri", "u", "p"), driver=mock_driver, own_driver=False)
    assert manager.verify_connection() is False
    mock_driver.session.assert_called_once()

def test_close_owned_driver(monkeypatch):
    mock_driver = mock.MagicMock()
    manager = Neo4jManager(settings=DummySettings("uri", "u", "p"), driver=mock_driver, own_driver=True)
    # Ensure driver is set
    _ = manager.driver
    manager.close()
    mock_driver.close.assert_called_once()
    # Subsequent close is safe
    manager.close()
    # close should not be called again
    assert mock_driver.close.call_count == 1

def test_close_injected_driver_not_owned(monkeypatch):
    mock_driver = mock.MagicMock()
    manager = Neo4jManager(settings=DummySettings("uri", "u", "p"), driver=mock_driver, own_driver=False)
    manager.close()
    mock_driver.close.assert_not_called()
