# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from unittest.mock import AsyncMock, MagicMock

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.inference import UserMessage, CompletionMessage
from llama_stack.core.memory.persistent_memory import (
    MemoryServiceImpl,
    PersistentMemoryConfig,
    PersistentMemoryStoreMeta,
)
from llama_stack.providers.utils.kvstore.kvstore import InmemoryKVStoreImpl


@pytest.fixture
def kv_store():
    """Create an in-memory KV store for testing."""
    return InmemoryKVStoreImpl()


@pytest.fixture
def vector_io_api():
    """Create a mock VectorIO API."""
    mock = AsyncMock()
    mock.insert = AsyncMock()
    mock.query_chunks = AsyncMock()
    return mock


@pytest.fixture
def inference_api():
    """Create a mock Inference API."""
    return AsyncMock()


@pytest.fixture
def memory_config():
    """Create a default memory configuration."""
    return PersistentMemoryConfig(
        enabled=True,
        provider_id="faiss",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        chunk_size_in_tokens=256,
        overlap_size_in_tokens=64,
    )


@pytest.fixture
async def memory_service(memory_config, kv_store, vector_io_api, inference_api):
    """Create a memory service instance."""
    await kv_store.initialize()
    service = MemoryServiceImpl(
        config=memory_config,
        kv_store=kv_store,
        vector_io_api=vector_io_api,
        inference_api=inference_api,
    )
    return service


@pytest.mark.asyncio
class TestMemoryService:
    async def test_create_store(self, memory_service, kv_store):
        """Test creating a new memory store."""
        agent_id = "test-agent-1"
        agent_config = AgentConfig(
            model="test-model",
            instructions="Test instructions",
        )
        
        store_meta = await memory_service.get_or_create_store(agent_id, agent_config)
        
        assert store_meta is not None
        assert store_meta.agent_id == agent_id
        assert store_meta.vector_db_id == f"mem-agent-{agent_id}"
        assert store_meta.embedding_model == memory_service.config.embedding_model
        assert store_meta.dimension == memory_service.config.embedding_dimension
        
        # Verify it was persisted
        retrieved_meta = await memory_service.get_store_meta(agent_id)
        assert retrieved_meta is not None
        assert retrieved_meta.agent_id == agent_id
        assert retrieved_meta.vector_db_id == store_meta.vector_db_id
    
    async def test_get_existing_store(self, memory_service):
        """Test retrieving an existing memory store."""
        agent_id = "test-agent-2"
        agent_config = AgentConfig(
            model="test-model",
            instructions="Test instructions",
        )
        
        # Create store
        store_meta_1 = await memory_service.get_or_create_store(agent_id, agent_config)
        
        # Get it again - should return the same store
        store_meta_2 = await memory_service.get_or_create_store(agent_id, agent_config)
        
        assert store_meta_1.id == store_meta_2.id
        assert store_meta_1.vector_db_id == store_meta_2.vector_db_id
    
    async def test_add_turn(self, memory_service, vector_io_api):
        """Test adding memory entries from a turn."""
        agent_id = "test-agent-3"
        session_id = "test-session-1"
        
        # Create store first
        agent_config = AgentConfig(
            model="test-model",
            instructions="Test instructions",
        )
        await memory_service.get_or_create_store(agent_id, agent_config)
        
        # Add a turn
        messages = [UserMessage(content="What is the capital of France?")]
        from llama_stack.apis.inference import StopReason
        output = CompletionMessage(
            content="The capital of France is Paris.",
            stop_reason=StopReason.end_of_turn,
        )
        
        await memory_service.add_turn(
            agent_id=agent_id,
            session_id=session_id,
            messages=messages,
            output=output,
        )
        
        # Verify that VectorIO insert was called
        assert vector_io_api.insert.called
        call_args = vector_io_api.insert.call_args
        assert call_args.kwargs["vector_db_id"] == f"mem-agent-{agent_id}"
        assert len(call_args.kwargs["documents"]) == 2  # One for user message, one for output
    
    async def test_search(self, memory_service, vector_io_api):
        """Test searching for memory entries."""
        agent_id = "test-agent-4"
        
        # Create store
        agent_config = AgentConfig(
            model="test-model",
            instructions="Test instructions",
        )
        await memory_service.get_or_create_store(agent_id, agent_config)
        
        # Mock search results
        from llama_stack.apis.vector_io import Chunk, ChunkMetadata, QueryChunksResponse
        
        mock_chunk = Chunk(
            content="Paris is the capital of France",
            metadata={"agent_id": agent_id},
            stored_chunk_id="chunk-1",
            chunk_metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_id="chunk-1",
                source="test",
                metadata_token_count=10,
            ),
        )
        vector_io_api.query_chunks.return_value = QueryChunksResponse(
            chunks=[mock_chunk],
            scores=[0.9],
        )
        
        # Search
        results = await memory_service.search(agent_id, "What is the capital?", max_chunks=5)
        
        assert len(results) == 1
        assert results[0]["content"] == "Paris is the capital of France"
        assert results[0]["metadata"]["agent_id"] == agent_id
    
    async def test_get_nonexistent_store(self, memory_service):
        """Test getting a store that doesn't exist."""
        agent_id = "nonexistent-agent"
        
        store_meta = await memory_service.get_store_meta(agent_id)
        
        assert store_meta is None
    
    async def test_custom_vector_db_id(self, memory_service):
        """Test creating a store with a custom vector_db_id."""
        agent_id = "test-agent-5"
        custom_vector_db_id = "custom-memory-store"
        
        agent_config = AgentConfig(
            model="test-model",
            instructions="Test instructions",
            persistent_memory={
                "enabled": True,
                "vector_db_id": custom_vector_db_id,
            },
        )
        
        store_meta = await memory_service.get_or_create_store(agent_id, agent_config)
        
        assert store_meta.vector_db_id == custom_vector_db_id
