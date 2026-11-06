import unittest
from unittest.mock import MagicMock, call

from graph.schema import GraphSchema, GraphSchemaError
from graph.connection import Neo4jManager

class TestGraphSchema(unittest.TestCase):
    def setUp(self):
        # Mock driver and session
        self.mock_session = MagicMock()
        self.mock_driver = MagicMock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session
        self.mock_manager = MagicMock(spec=Neo4jManager)
        self.mock_manager.driver = self.mock_driver
        self.schema = GraphSchema(self.mock_manager)

    def test_node_labels_and_relationship_types(self):
        self.assertCountEqual(
            self.schema.node_labels,
            ["Paper", "Concept", "Method", "Dataset"],
        )
        self.assertCountEqual(
            self.schema.relationship_types,
            ["USES_METHOD", "EVALUATES_ON", "RELATED_TO"],
        )

    def test_constraints_and_indexes_content(self):
        constraints = self.schema._constraints()
        indexes = self.schema._indexes()
        # Verify count
        self.assertEqual(len(constraints), 4)
        self.assertEqual(len(indexes), 4)
        # Verify exact constraint statements
        expected_constraints = [
            f"CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:{self.schema.PAPER}) REQUIRE p.{self.schema.PROP_PAPER_ID} IS UNIQUE",
            f"CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:{self.schema.CONCEPT}) REQUIRE c.{self.schema.PROP_CONCEPT_ID} IS UNIQUE",
            f"CREATE CONSTRAINT method_id_unique IF NOT EXISTS FOR (m:{self.schema.METHOD}) REQUIRE m.{self.schema.PROP_METHOD_ID} IS UNIQUE",
            f"CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS FOR (d:{self.schema.DATASET}) REQUIRE d.{self.schema.PROP_DATASET_ID} IS UNIQUE",
        ]
        self.assertEqual(constraints, expected_constraints)

        # Verify exact index statements
        expected_indexes = [
            f"CREATE INDEX IF NOT EXISTS FOR (p:{self.schema.PAPER}) ON (p.title)",
            f"CREATE INDEX IF NOT EXISTS FOR (c:{self.schema.CONCEPT}) ON (c.name)",
            f"CREATE INDEX IF NOT EXISTS FOR (m:{self.schema.METHOD}) ON (m.name)",
            f"CREATE INDEX IF NOT EXISTS FOR (d:{self.schema.DATASET}) ON (d.name)",
        ]
        self.assertEqual(indexes, expected_indexes)

    def test_all_statements_executed(self):
        self.schema.create_schema()
        expected_calls = [call(stmt) for stmt in self.schema._constraints() + self.schema._indexes()]
        self.assertEqual(self.mock_session.run.call_args_list, expected_calls)

    def test_idempotent_execution(self):
        # First execution
        self.schema.create_schema()
        first_calls = self.mock_session.run.call_args_list.copy()
        self.mock_session.run.reset_mock()
        # Second execution should run the same statements again without error
        self.schema.create_schema()
        second_calls = self.mock_session.run.call_args_list
        self.assertEqual(first_calls, second_calls)
        self.assertTrue(self.mock_session.run.called)

    def test_error_wrapped_in_graph_schema_error(self):
        self.mock_session.run.side_effect = Exception("Cypher error")
        with self.assertRaises(GraphSchemaError) as ctx:
            self.schema.create_schema()
        self.assertIn("Cypher error", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
