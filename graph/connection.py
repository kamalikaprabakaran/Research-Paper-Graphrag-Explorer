"""graph/connection.py

Provides a lightweight, test-friendly wrapper around the official Neo4j Python driver.
The manager lazily creates a driver, validates configuration, offers a simple
`verify_connection` health-check, and handles safe cleanup.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checking; at runtime we import lazily.
    from neo4j import Driver

from config.settings import Settings


class Neo4jConfigurationError(RuntimeError):
    """Raised when required Neo4j configuration values are missing or empty."""

    def __init__(self, missing: list[str]):
        message = f"Missing Neo4j configuration for: {', '.join(missing)}"
        super().__init__(message)
        self.missing = missing


class Neo4jManager:
    """Manage a Neo4j driver instance with lazy creation and optional injection.

    Parameters
    ----------
    settings: Settings | None, optional
        Application settings. If omitted, a default ``Settings()`` instance is used.
    driver: Driver | None, optional
        An already-constructed Neo4j driver (e.g., a mock in tests). When provided,
        the manager will **not** create its own driver and will treat the driver as
        externally owned - ``close()`` will not close it unless ``own_driver`` is set
        explicitly.
    own_driver: bool, default ``True`` when ``driver`` is ``None``
        Indicates whether the manager is responsible for closing the driver.
    """

    def __init__(self, settings: Settings | None = None, driver: Optional["Driver"] = None, *, own_driver: bool = True) -> None:
        self._settings = settings or Settings()
        self._driver: Optional["Driver"] = driver
        # Respect caller's ownership flag when a driver is injected.
        self._own_driver = own_driver if driver is not None else True

    @property
    def driver(self) -> "Driver":
        """Return a live ``neo4j.Driver`` instance, creating it on first use.

        Raises
        ------
        Neo4jConfigurationError
            If any of ``NEO4J_URI``, ``NEO4J_USERNAME`` or ``NEO4J_PASSWORD`` are missing or empty.
        """
        if self._driver is None:
            # Lazy import to avoid ImportError at module load time.
            try:
                from neo4j import GraphDatabase
            except ImportError as e:
                raise ImportError("neo4j package is required for Neo4jManager but is not installed") from e

            missing: list[str] = []
            uri = self._settings.NEO4J_URI
            user = self._settings.NEO4J_USERNAME
            pwd = self._settings.NEO4J_PASSWORD
            if not uri:
                missing.append("NEO4J_URI")
            if not user:
                missing.append("NEO4J_USERNAME")
            if not pwd:
                missing.append("NEO4J_PASSWORD")
            if missing:
                raise Neo4jConfigurationError(missing)

            self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._own_driver = True
        return self._driver

    def verify_connection(self) -> bool:
        """Execute a lightweight ``RETURN 1`` query to confirm connectivity.

        Returns ``True`` on success, ``False`` on any exception.
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the underlying driver if this manager owns it.

        The method is idempotent - calling it multiple times is safe.
        """
        if self._driver is not None and self._own_driver:
            try:
                self._driver.close()
            finally:
                self._driver = None
                self._own_driver = False
