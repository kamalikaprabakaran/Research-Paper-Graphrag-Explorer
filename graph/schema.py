"""graph/schema.py

Defines the Neo4j knowledge graph schema, including node labels, relationship types,
uniqueness constraints, and indexes. Uses a provided ``Neo4jManager`` for driver
access.
"""

import logging
from typing import Any, List
from .connection import Neo4jManager


class GraphSchemaError(RuntimeError):
    """Raised when schema creation fails."""

    def __init__(self, message: str):
        super().__init__(message)


class GraphSchema:
    """Define and manage the Neo4j knowledge graph schema.

    This class uses an existing :class:`Neo4jManager` instance for database
    interactions. It defines node labels, relationship types, uniqueness
    constraints, and indexes. The :meth:`create_schema` method is idempotent -
    constraints and indexes are created with ``IF NOT EXISTS`` so repeated
    calls are safe.
    """

    # Node labels
    PAPER = "Paper"
    CONCEPT = "Concept"
    METHOD = "Method"
    DATASET = "Dataset"

    # Relationship types
    USES_METHOD = "USES_METHOD"
    EVALUATES_ON = "EVALUATES_ON"
    RELATED_TO = "RELATED_TO"
    # Provenance properties for relationships (documented for future ingestion)
    # - source_paper_id: ID of the originating paper
    # - source_page: page number within the source paper
    # - source_chunk_id: identifier of the text chunk
    # - confidence: confidence score of the extracted relationship

    # Property names (stable identifiers)
    PROP_PAPER_ID = "paper_id"
    PROP_CONCEPT_ID = "concept_id"
    PROP_METHOD_ID = "method_id"
    PROP_DATASET_ID = "dataset_id"

    def __init__(self, manager: Neo4jManager):
        self._manager = manager
        self._logger = logging.getLogger(self.__class__.__name__)

    # ---------------------------------------------------------------------
    # Internal helpers - constraints & indexes
    # ---------------------------------------------------------------------
    def _constraints(self) -> List[str]:
        """Return Cypher statements for uniqueness constraints.

        All constraints use ``IF NOT EXISTS`` for idempotent execution.
        """
        return [
            f"CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:{self.PAPER}) REQUIRE p.{self.PROP_PAPER_ID} IS UNIQUE",
            f"CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:{self.CONCEPT}) REQUIRE c.{self.PROP_CONCEPT_ID} IS UNIQUE",
            f"CREATE CONSTRAINT method_id_unique IF NOT EXISTS FOR (m:{self.METHOD}) REQUIRE m.{self.PROP_METHOD_ID} IS UNIQUE",
            f"CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS FOR (d:{self.DATASET}) REQUIRE d.{self.PROP_DATASET_ID} IS UNIQUE",
        ]

    def _indexes(self) -> List[str]:
        """Return Cypher statements for useful indexes.

        Indexes are also created with ``IF NOT EXISTS``.
        """
        return [
            f"CREATE INDEX IF NOT EXISTS FOR (p:{self.PAPER}) ON (p.title)",
            f"CREATE INDEX IF NOT EXISTS FOR (c:{self.CONCEPT}) ON (c.name)",
            f"CREATE INDEX IF NOT EXISTS FOR (m:{self.METHOD}) ON (m.name)",
            f"CREATE INDEX IF NOT EXISTS FOR (d:{self.DATASET}) ON (d.name)",
        ]

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def create_schema(self) -> None:
        """Create constraints and indexes for the knowledge graph.

        The method opens a Neo4j session and executes each constraint and index
        statement individually via ``session.run``. This maintains idempotency
        because the Cypher statements use ``IF NOT EXISTS``. Any exception is
        wrapped in :class:`GraphSchemaError`.
        """
        try:
            driver: Any = self._manager.driver
            with driver.session() as session:
                for stmt in self._constraints():
                    self._logger.debug("Creating constraint: %s", stmt)
                    session.run(stmt)
                for stmt in self._indexes():
                    self._logger.debug("Creating index: %s", stmt)
                    session.run(stmt)
        except Exception as exc:
            self._logger.error("Failed to create graph schema: %s", exc)
            raise GraphSchemaError(str(exc)) from exc

    # ---------------------------------------------------------------------
    # Helper for external introspection (optional)
    # ---------------------------------------------------------------------
    @property
    def node_labels(self) -> List[str]:
        return [self.PAPER, self.CONCEPT, self.METHOD, self.DATASET]

    @property
    def relationship_types(self) -> List[str]:
        return [self.USES_METHOD, self.EVALUATES_ON, self.RELATED_TO]
