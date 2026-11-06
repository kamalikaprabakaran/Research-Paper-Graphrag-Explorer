# Research Paper Connection Explorer

## Overview

A modular Python project for ingesting research papers, building a knowledge graph, and generating evidence-grounded answers using hybrid GraphRAG techniques.

## Technology Stack

- **Python 3.11/3.12** - Core programming language
- **Streamlit** - User interface
- **PyMuPDF** - PDF text extraction
- **Sentence Transformers** - Text embeddings
- **FAISS CPU** - Vector similarity search
- **Neo4j Aura** - Knowledge graph storage
- **OpenRouter API** - LLM provider
- **PyVis** - Knowledge graph visualization
- **Pydantic** - Configuration and data validation
- **python-dotenv** - Environment variable management
- **Pytest** - Automated testing

## Project Structure

```text
GraphRAG/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py
├── ingestion/
│   └── __init__.py
├── graph/
│   └── __init__.py
├── retrieval/
│   └── __init__.py
├── generation/
│   └── __init__.py
├── evaluation/
│   └── __init__.py
├── database/
│   └── __init__.py
├── tests/
│   ├── __init__.py
│   └── test_config.py
└── data/
    └── .gitkeep
```

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env` and add your OpenRouter API key and Neo4j Aura credentials when those integrations are implemented.

Never commit `.env` or expose API keys in source code.

## Running the Application

```powershell
streamlit run app.py
```

During Phase 0, the application displays the project overview and planned pipeline. External API and database connections are not required.

## Running Tests

```powershell
python -m pytest -v
```

The initial test suite checks configuration initialization and environment variable loading without requiring live external services.

## Development Roadmap

1. **Phase 0:** Project foundation and configuration - completed.
2. **Phase 1:** PDF ingestion and metadata extraction
3. **Phase 2:** Text chunking, embeddings, vector search, and baseline RAG
4. **Phase 3:** Knowledge graph construction using Neo4j Aura
5. **Phase 4:** Graph-based retrieval
6. **Phase 5:** Hybrid GraphRAG retrieval and answer generation
7. **Phase 6:** Research chat, graph visualization, and paper comparison UI
8. **Phase 7:** Testing and evaluation of retrieval quality and answer grounding

## Current Status

**Phase 0: Project foundation and environment setup - completed.**

The initial configuration tests pass. PDF ingestion, vector retrieval, knowledge graph construction, and answer generation will be implemented in subsequent phases.

## Phase 3.2 Knowledge Graph Schema

The graph schema defines four node labels:

- **Paper** - unique identifier `paper_id`, indexed `title`.
- **Concept** - unique identifier `concept_id`, indexed `name`.
- **Method** - unique identifier `method_id`, indexed `name`.
- **Dataset** - unique identifier `dataset_id`, indexed `name`.

Relationship types:

- `USES_METHOD` (Paper -> Method)
- `EVALUATES_ON` (Paper -> Dataset)
- `RELATED_TO` (Paper <-> Concept, Concept <-> Concept, Method <-> Method)

Uniqueness constraints (4):

```cypher
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE
CREATE CONSTRAINT concept_id_unique IF NOT EXISTS FOR (c:Concept) REQUIRE c.concept_id IS UNIQUE
CREATE CONSTRAINT method_id_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.method_id IS UNIQUE
CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS FOR (d:Dataset) REQUIRE d.dataset_id IS UNIQUE
```

Indexes (4):

```cypher
CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title)
CREATE INDEX IF NOT EXISTS FOR (c:Concept) ON (c.name)
CREATE INDEX IF NOT EXISTS FOR (m:Method) ON (m.name)
CREATE INDEX IF NOT EXISTS FOR (d:Dataset) ON (d.name)
```

The schema is initialized by calling `GraphSchema(Neo4jManager()).create_schema()` after a `Neo4jManager` instance is configured.

Required Neo4j environment variables (set in `.env` but not included here):

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
