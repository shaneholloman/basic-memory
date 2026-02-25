"""Tests for ProjectService.get_embedding_status()."""

from unittest.mock import patch

import pytest

from basic_memory.schemas.project_info import EmbeddingStatus
from basic_memory.services.project_service import ProjectService


@pytest.mark.asyncio
async def test_embedding_status_semantic_disabled(project_service: ProjectService, test_project):
    """When semantic search is disabled, return minimal status with zero counts."""
    with patch.object(
        type(project_service),
        "config_manager",
        new_callable=lambda: property(
            lambda self: _config_manager_with(semantic_search_enabled=False)
        ),
    ):
        status = await project_service.get_embedding_status(test_project.id)

    assert isinstance(status, EmbeddingStatus)
    assert status.semantic_search_enabled is False
    assert status.reindex_recommended is False
    assert status.total_chunks == 0
    assert status.total_embeddings == 0


@pytest.mark.asyncio
async def test_embedding_status_vector_tables_missing(
    project_service: ProjectService, test_graph, test_project
):
    """When vector tables don't exist, recommend reindex."""
    with patch.object(
        type(project_service),
        "config_manager",
        new_callable=lambda: property(
            lambda self: _config_manager_with(semantic_search_enabled=True)
        ),
    ):
        status = await project_service.get_embedding_status(test_project.id)

    # Vector tables are not created by the standard test fixtures
    # If they don't exist, status should flag it
    assert status.semantic_search_enabled is True
    assert status.embedding_provider == "fastembed"
    assert status.embedding_model == "bge-small-en-v1.5"

    if not status.vector_tables_exist:
        assert status.reindex_recommended is True
        assert "Vector tables not initialized" in (status.reindex_reason or "")


@pytest.mark.asyncio
async def test_embedding_status_entities_without_chunks(
    project_service: ProjectService, test_graph, test_project
):
    """When entities have search_index rows but no chunks, recommend reindex."""
    # Create vector tables (empty) so the table-existence check passes
    from sqlalchemy import text

    await project_service.repository.execute_query(
        text(
            "CREATE TABLE IF NOT EXISTS search_vector_chunks ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  entity_id INTEGER NOT NULL,"
            "  project_id INTEGER NOT NULL,"
            "  chunk_key TEXT NOT NULL,"
            "  chunk_text TEXT NOT NULL,"
            "  source_hash TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        ),
        {},
    )

    with patch.object(
        type(project_service),
        "config_manager",
        new_callable=lambda: property(
            lambda self: _config_manager_with(semantic_search_enabled=True)
        ),
    ):
        status = await project_service.get_embedding_status(test_project.id)

    assert status.semantic_search_enabled is True
    assert status.vector_tables_exist is True
    # test_graph creates entities indexed in search_index, but no vector chunks
    assert status.total_indexed_entities > 0
    assert status.total_chunks == 0
    assert status.reindex_recommended is True
    assert "never been built" in (status.reindex_reason or "")


@pytest.mark.asyncio
async def test_embedding_status_orphaned_chunks(
    project_service: ProjectService, test_graph, test_project
):
    """When chunks exist without matching embeddings, recommend reindex."""
    from sqlalchemy import text

    # Create vector tables
    await project_service.repository.execute_query(
        text(
            "CREATE TABLE IF NOT EXISTS search_vector_chunks ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  entity_id INTEGER NOT NULL,"
            "  project_id INTEGER NOT NULL,"
            "  chunk_key TEXT NOT NULL,"
            "  chunk_text TEXT NOT NULL,"
            "  source_hash TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        ),
        {},
    )

    # Insert a chunk row (no matching embedding = orphan)
    # Get a real entity_id from the test graph
    entity_result = await project_service.repository.execute_query(
        text("SELECT id FROM entity WHERE project_id = :project_id LIMIT 1"),
        {"project_id": test_project.id},
    )
    entity_id = entity_result.scalar()

    await project_service.repository.execute_query(
        text(
            "INSERT INTO search_vector_chunks "
            "(entity_id, project_id, chunk_key, chunk_text, source_hash) "
            "VALUES (:entity_id, :project_id, 'chunk-1', 'test text', 'abc123')"
        ),
        {"entity_id": entity_id, "project_id": test_project.id},
    )

    # Create a minimal search_vector_embeddings table (not sqlite-vec virtual table)
    # so the LEFT JOIN works and finds the orphan
    await project_service.repository.execute_query(
        text(
            "CREATE TABLE IF NOT EXISTS search_vector_embeddings ("
            "  rowid INTEGER PRIMARY KEY"
            ")"
        ),
        {},
    )

    with patch.object(
        type(project_service),
        "config_manager",
        new_callable=lambda: property(
            lambda self: _config_manager_with(semantic_search_enabled=True)
        ),
    ):
        status = await project_service.get_embedding_status(test_project.id)

    assert status.vector_tables_exist is True
    assert status.total_chunks == 1
    assert status.orphaned_chunks == 1
    assert status.reindex_recommended is True
    assert "orphaned chunks" in (status.reindex_reason or "")


@pytest.mark.asyncio
async def test_embedding_status_healthy(
    project_service: ProjectService, test_graph, test_project
):
    """When all entities have embeddings, no reindex recommended."""
    from sqlalchemy import text

    # Create vector chunks table
    await project_service.repository.execute_query(
        text(
            "CREATE TABLE IF NOT EXISTS search_vector_chunks ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  entity_id INTEGER NOT NULL,"
            "  project_id INTEGER NOT NULL,"
            "  chunk_key TEXT NOT NULL,"
            "  chunk_text TEXT NOT NULL,"
            "  source_hash TEXT NOT NULL,"
            "  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        ),
        {},
    )

    # Drop any existing virtual table (may have been created by search_service init)
    # and recreate as a simple regular table for testing the join logic
    await project_service.repository.execute_query(
        text("DROP TABLE IF EXISTS search_vector_embeddings"), {}
    )
    await project_service.repository.execute_query(
        text(
            "CREATE TABLE search_vector_embeddings ("
            "  rowid INTEGER PRIMARY KEY"
            ")"
        ),
        {},
    )

    # Insert a chunk + matching embedding for every search_index entity
    entity_result = await project_service.repository.execute_query(
        text(
            "SELECT DISTINCT entity_id FROM search_index WHERE project_id = :project_id"
        ),
        {"project_id": test_project.id},
    )
    entity_ids = [row[0] for row in entity_result.fetchall()]

    chunk_id = 1
    for eid in entity_ids:
        await project_service.repository.execute_query(
            text(
                "INSERT INTO search_vector_chunks "
                "(id, entity_id, project_id, chunk_key, chunk_text, source_hash) "
                "VALUES (:id, :entity_id, :project_id, :key, 'text', 'hash')"
            ),
            {
                "id": chunk_id,
                "entity_id": eid,
                "project_id": test_project.id,
                "key": f"chunk-{chunk_id}",
            },
        )
        await project_service.repository.execute_query(
            text("INSERT INTO search_vector_embeddings (rowid) VALUES (:rowid)"),
            {"rowid": chunk_id},
        )
        chunk_id += 1

    with patch.object(
        type(project_service),
        "config_manager",
        new_callable=lambda: property(
            lambda self: _config_manager_with(semantic_search_enabled=True)
        ),
    ):
        status = await project_service.get_embedding_status(test_project.id)

    assert status.vector_tables_exist is True
    assert status.total_chunks > 0
    assert status.total_embeddings == status.total_chunks
    assert status.orphaned_chunks == 0
    assert status.reindex_recommended is False
    assert status.reindex_reason is None


@pytest.mark.asyncio
async def test_get_project_info_includes_embedding_status(
    project_service: ProjectService, test_graph, test_project
):
    """get_project_info() response includes embedding_status field."""
    info = await project_service.get_project_info(test_project.name)
    assert info.embedding_status is not None
    assert isinstance(info.embedding_status, EmbeddingStatus)


# --- Helper ---


def _config_manager_with(semantic_search_enabled: bool):
    """Create a ConfigManager whose config has the given semantic_search_enabled value."""
    from basic_memory.config import ConfigManager

    cm = ConfigManager()
    # Patch the config object in-place
    cm.config.semantic_search_enabled = semantic_search_enabled
    return cm
