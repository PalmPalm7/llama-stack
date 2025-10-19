# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import Chunk, ChunkMetadata, QueryChunksResponse
from llama_stack.providers.remote.vector_io.mem0.config import Mem0VectorIOConfig
from llama_stack.providers.remote.vector_io.mem0.mem0 import Mem0VectorIOImpl

# This test is a unit test for the Mem0VectorIOImpl class. This should only contain
# tests which are specific to this class. More general (API-level) tests should be placed in
# tests/integration/vector_io/
#
# How to run this test:
#
# pytest tests/unit/providers/vector_io/test_mem0.py \
# -v -s --tb=short --disable-warnings --asyncio-mode=auto


@pytest.fixture(scope="session")
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def embedding_dimension():
    return 384


@pytest.fixture
def vector_db_id():
    return "test_vector_db"


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            content="Test content 1",
            metadata={"document_id": "doc-1"},
            chunk_metadata=ChunkMetadata(
                chunk_id="chunk-1", document_id="doc-1", source="test"
            ),
        ),
        Chunk(
            content="Test content 2",
            metadata={"document_id": "doc-2"},
            chunk_metadata=ChunkMetadata(
                chunk_id="chunk-2", document_id="doc-2", source="test"
            ),
        ),
    ]


@pytest.fixture
def mock_vector_db(vector_db_id, embedding_dimension) -> VectorDB:
    return VectorDB(
        identifier=vector_db_id,
        provider_id="mem0",
        embedding_model="mock_embedding_model",
        embedding_dimension=embedding_dimension,
        provider_resource_id=None,
    )


@pytest.fixture
def mock_files_api():
    return MagicMock(spec=Files)


@pytest.fixture
def mock_inference_api():
    return MagicMock(spec=Inference)


@pytest.fixture
def mem0_cloud_config():
    return Mem0VectorIOConfig(
        is_cloud=True,
        api_key="test_api_key",
        user_namespace="test_ns",
        default_limit=10,
    )


@pytest.fixture
def mem0_local_config():
    return Mem0VectorIOConfig(
        is_cloud=False,
        local_config={
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}},
            "embedder": {
                "provider": "openai",
                "config": {"model": "text-embedding-3-small"},
            },
        },
        user_namespace="test_ns",
        default_limit=10,
    )


@pytest.fixture
def mock_mem0_cloud_client():
    client = MagicMock()
    client.add = MagicMock(return_value={"id": "test-memory-id"})
    client.search = MagicMock(
        return_value={
            "results": [
                {
                    "memory": "Test content 1",
                    "metadata": {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-1",
                        "vector_db_id": "test_vector_db",
                    },
                    "score": 0.95,
                }
            ]
        }
    )
    client.delete_all = MagicMock()
    return client


@pytest.fixture
def mock_mem0_local_client():
    client = MagicMock()
    client.__class__.__name__ = "Memory"
    client.add = MagicMock(return_value={"id": "test-memory-id"})
    client.search = MagicMock(
        return_value=[
            {
                "memory": "Test content 1",
                "metadata": {
                    "chunk_id": "chunk-1",
                    "document_id": "doc-1",
                    "vector_db_id": "test_vector_db",
                },
                "score": 0.95,
            }
        ]
    )
    client.delete_all = MagicMock()
    return client


async def test_mem0_cloud_initialization(
    mem0_cloud_config, mock_inference_api, mock_files_api
):
    """Test that cloud client is initialized correctly."""
    with patch(
        "llama_stack.providers.remote.vector_io.mem0.mem0.Mem0CloudClient"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        impl = Mem0VectorIOImpl(
            config=mem0_cloud_config,
            inference_api=mock_inference_api,
            files_api=mock_files_api,
        )
        await impl.initialize()

        mock_client_class.assert_called_once_with(api_key="test_api_key")
        assert impl._client == mock_client


async def test_mem0_local_initialization(
    mem0_local_config, mock_inference_api, mock_files_api
):
    """Test that local client is initialized correctly."""
    with patch(
        "llama_stack.providers.remote.vector_io.mem0.mem0.Mem0Local"
    ) as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.from_config.return_value = mock_client

        impl = Mem0VectorIOImpl(
            config=mem0_local_config,
            inference_api=mock_inference_api,
            files_api=mock_files_api,
        )
        await impl.initialize()

        mock_client_class.from_config.assert_called_once()
        assert impl._client == mock_client


async def test_register_vector_db(
    mem0_cloud_config, mock_inference_api, mock_files_api, mock_vector_db
):
    """Test registering a vector database."""
    impl = Mem0VectorIOImpl(
        config=mem0_cloud_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )

    await impl.register_vector_db(mock_vector_db)

    assert mock_vector_db.identifier in impl._vector_dbs
    assert impl._vector_dbs[mock_vector_db.identifier] == mock_vector_db


async def test_insert_chunks_cloud(
    mem0_cloud_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_cloud_client,
    vector_db_id,
    sample_chunks,
):
    """Test inserting chunks with cloud client."""
    impl = Mem0VectorIOImpl(
        config=mem0_cloud_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    impl._client = mock_mem0_cloud_client

    await impl.insert_chunks(vector_db_id=vector_db_id, chunks=sample_chunks)

    # Verify add was called for each chunk
    assert mock_mem0_cloud_client.add.call_count == len(sample_chunks)


async def test_insert_chunks_local(
    mem0_local_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_local_client,
    vector_db_id,
    sample_chunks,
):
    """Test inserting chunks with local client."""
    impl = Mem0VectorIOImpl(
        config=mem0_local_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    impl._client = mock_mem0_local_client

    await impl.insert_chunks(vector_db_id=vector_db_id, chunks=sample_chunks)

    # Verify add was called for each chunk
    assert mock_mem0_local_client.add.call_count == len(sample_chunks)


async def test_query_chunks_cloud(
    mem0_cloud_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_cloud_client,
    vector_db_id,
):
    """Test querying chunks with cloud client."""
    impl = Mem0VectorIOImpl(
        config=mem0_cloud_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    impl._client = mock_mem0_cloud_client

    response = await impl.query_chunks(
        vector_db_id=vector_db_id, query="test query", params={"max_chunks": 5}
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 1
    assert len(response.scores) == 1
    assert response.chunks[0].content == "Test content 1"
    assert response.scores[0] == 0.95


async def test_query_chunks_local(
    mem0_local_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_local_client,
    vector_db_id,
):
    """Test querying chunks with local client."""
    impl = Mem0VectorIOImpl(
        config=mem0_local_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    impl._client = mock_mem0_local_client

    response = await impl.query_chunks(
        vector_db_id=vector_db_id, query="test query", params={"max_chunks": 5}
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 1
    assert len(response.scores) == 1
    assert response.chunks[0].content == "Test content 1"
    assert response.scores[0] == 0.95


async def test_unregister_vector_db(
    mem0_cloud_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_cloud_client,
    mock_vector_db,
    vector_db_id,
):
    """Test unregistering a vector database."""
    impl = Mem0VectorIOImpl(
        config=mem0_cloud_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    impl._client = mock_mem0_cloud_client

    # Register first
    await impl.register_vector_db(mock_vector_db)
    assert vector_db_id in impl._vector_dbs

    # Unregister
    await impl.unregister_vector_db(vector_db_id)

    assert vector_db_id not in impl._vector_dbs
    mock_mem0_cloud_client.delete_all.assert_called_once()


async def test_delete_chunks_not_implemented(
    mem0_cloud_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_cloud_client,
    vector_db_id,
):
    """Test that delete_chunks raises NotImplementedError."""
    impl = Mem0VectorIOImpl(
        config=mem0_cloud_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    impl._client = mock_mem0_cloud_client

    with pytest.raises(NotImplementedError):
        await impl.delete_chunks(vector_db_id=vector_db_id, chunk_ids=["chunk-1"])


async def test_query_chunks_handles_error(
    mem0_cloud_config,
    mock_inference_api,
    mock_files_api,
    mock_mem0_cloud_client,
    vector_db_id,
):
    """Test that query_chunks handles errors gracefully."""
    impl = Mem0VectorIOImpl(
        config=mem0_cloud_config,
        inference_api=mock_inference_api,
        files_api=mock_files_api,
    )
    mock_mem0_cloud_client.search.side_effect = Exception("Search error")
    impl._client = mock_mem0_cloud_client

    response = await impl.query_chunks(
        vector_db_id=vector_db_id, query="test query", params={"max_chunks": 5}
    )

    assert isinstance(response, QueryChunksResponse)
    assert len(response.chunks) == 0
    assert len(response.scores) == 0
