"""graph/connection.py

Provides a lightweight, test‑friendly wrapper around the official Neo4j Python driver.
The manager lazily creates a driver, validates configuration, offers a simple
`verify_connection` health‑check, and handles safe cleanup.
"""

from __future__ import annotations

from typing import Optional

from neo4j import GraphDatabase, Driver

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
        An already‑constructed Neo4j driver (e.g., a mock in tests). When provided,
        the manager will **not** create its own driver and will treat the driver as
        externally owned – ``close()`` will not close it unless ``own_driver`` is set
        explicitly.
    own_driver: bool, default ``True`` when ``driver`` is ``None``
        Indicates whether the manager is responsible for closing the driver. If a
        driver is injected and ``own_driver`` is ``False`` (default), ``close()``
        leaves the driver untouched.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        driver: Optional[Driver] = None,
        *,
        own_driver: bool = True,
    ) -> None:
        self._settings = settings or Settings()
        self._driver: Optional[Driver] = driver
        # Ownership: if we created the driver we own it; if a driver is injected we
        # own it only when the caller explicitly requests it.
        self._own_driver = own_driver if driver is not None else True

    # ---------------------------------------------------------------------
    # Driver property – lazy creation
    # ---------------------------------------------------------------------
    @property
    def driver(self) -> Driver:
        """Return a live ``neo4j.Driver`` instance, creating it on first use.

        Raises
        ------
        Neo4jConfigurationError
            If any of ``NEO4J_URI``, ``NEO4J_USERNAME`` or ``NEO4J_PASSWORD`` are
            missing or empty.
        """
        if self._driver is None:
            missing = []
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
            # Create the driver lazily – no network call is made until a session
            # is opened.
            self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._own_driver = True
        return self._driver

    # ---------------------------------------------------------------------
    # Connection verification
    # ---------------------------------------------------------------------
    def verify_connection(self) -> bool:
        """Execute a lightweight ``RETURN 1`` query to confirm connectivity.

        Returns ``True`` on success, ``False`` on any exception. Errors are logged
        with a generic message to avoid leaking credentials.
        """
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1")
                # Consume the result to surface possible query errors.
                result.single()
            return True
        except Exception:
            # In a real application you would log the exception safely.
            return False

    # ---------------------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying driver if this manager owns it.

        The method is idempotent – calling it multiple times is safe.
        """
        if self._driver is not None and self._own_driver:
            try:
                self._driver.close()
            finally:
                # Ensure the reference is cleared even if close() raises.
                self._driver = None
                self._own_driver = False
