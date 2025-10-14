# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
import uuid
from datetime import UTC, datetime
from typing import Optional, Protocol

from pydantic import BaseModel

from llama_stack.apis.agents import AgentConfig
from llama_stack.apis.inference import Inference, Message, UserMessage
from llama_stack.apis.tools import RAGDocument
from llama_stack.apis.vector_dbs import VectorDB, VectorDBInput
from llama_stack.apis.vector_io import (
    VectorIO,
    VectorStoreChunkingStrategyStatic,
    VectorStoreChunkingStrategyStaticConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.kvstore.api import KVStore

logger = get_logger(name=__name__, category="memory")


class PersistentMemoryStoreMeta(BaseModel):
    """Metadata for a persistent memory store.
    
    :param id: Unique identifier for the memory store
    :param agent_id: ID of the agent this store belongs to
    :param user_id: Optional user ID for user-scoped memory
    :param vector_db_id: ID of the vector database backing this store
    :param created_at: Timestamp when the store was created
    :param embedding_model: Name of the embedding model used
    :param dimension: Dimension of embeddings
    :param chunk_size_in_tokens: Size of text chunks in tokens
    :param overlap_size_in_tokens: Overlap between chunks in tokens
    :param provider_id: ID of the vector DB provider
    """
    id: str
    agent_id: str
    user_id: str | None = None
    vector_db_id: str
    created_at: str
    embedding_model: str
    dimension: int
    chunk_size_in_tokens: int
    overlap_size_in_tokens: int
    provider_id: str


class MemoryEntry(BaseModel):
    """A single memory entry.
    
    :param id: Unique identifier for the entry
    :param agent_id: ID of the agent
    :param session_id: ID of the session
    :param role: Role of the message (user/assistant)
    :param content: Content of the message
    :param created_at: Timestamp when the entry was created
    :param metadata: Additional metadata
    """
    id: str
    agent_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    metadata: dict[str, str] | None = None


class PersistentMemoryConfig(BaseModel):
    """Configuration for persistent memory service.
    
    :param enabled: Whether persistent memory is enabled
    :param provider_id: Provider ID for vector DB (e.g., 'faiss')
    :param embedding_model: Name of the embedding model
    :param embedding_dimension: Dimension of embeddings
    :param chunk_size_in_tokens: Size of text chunks in tokens
    :param overlap_size_in_tokens: Overlap between chunks in tokens
    :param auto_summarize_turns: Whether to auto-summarize turns
    :param summary_model: Model to use for summarization
    """
    enabled: bool = False
    provider_id: str = "faiss"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    chunk_size_in_tokens: int = 256
    overlap_size_in_tokens: int = 64
    auto_summarize_turns: bool = False
    summary_model: str | None = None


class AgentPersistentMemoryConfig(BaseModel):
    """Agent-level persistent memory configuration.
    
    :param enabled: Whether persistent memory is enabled for this agent
    :param vector_db_id: Optional explicit vector_db_id to use
    """
    enabled: bool | None = None
    vector_db_id: str | None = None


class MemoryService(Protocol):
    """Service interface for managing persistent memory."""
    
    async def get_or_create_store(
        self, agent_id: str, agent_cfg: AgentConfig
    ) -> PersistentMemoryStoreMeta:
        """Get or create a memory store for an agent.
        
        :param agent_id: ID of the agent
        :param agent_cfg: Configuration of the agent
        :return: Metadata for the memory store
        """
        ...
    
    async def add_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: list[Message],
        output: Message,
        summarize: bool = False,
    ) -> None:
        """Add memory entries from a turn.
        
        :param agent_id: ID of the agent
        :param session_id: ID of the session
        :param messages: Input messages for the turn
        :param output: Output message from the turn
        :param summarize: Whether to summarize the turn
        """
        ...
    
    async def search(
        self, agent_id: str, query: str, max_chunks: int = 5
    ) -> list[dict]:
        """Search for relevant memory entries.
        
        :param agent_id: ID of the agent
        :param query: Query string
        :param max_chunks: Maximum number of chunks to return
        :return: List of relevant documents
        """
        ...
    
    async def get_store_meta(
        self, agent_id: str
    ) -> Optional[PersistentMemoryStoreMeta]:
        """Get metadata for an agent's memory store.
        
        :param agent_id: ID of the agent
        :return: Metadata or None if not found
        """
        ...


class MemoryServiceImpl:
    """Implementation of the MemoryService."""
    
    def __init__(
        self,
        config: PersistentMemoryConfig,
        kv_store: KVStore,
        vector_io_api: VectorIO,
        inference_api: Inference,
    ):
        self.config = config
        self.kv_store = kv_store
        self.vector_io_api = vector_io_api
        self.inference_api = inference_api
        self._store_prefix = "pm:store:"
    
    async def get_or_create_store(
        self, agent_id: str, agent_cfg: AgentConfig
    ) -> PersistentMemoryStoreMeta:
        """Get or create a memory store for an agent."""
        # Check if store already exists
        store_meta = await self.get_store_meta(agent_id)
        if store_meta:
            return store_meta
        
        # Create new store
        vector_db_id = f"mem-agent-{agent_id}"
        
        # Check if agent config has explicit vector_db_id
        if hasattr(agent_cfg, "persistent_memory") and agent_cfg.persistent_memory:
            if isinstance(agent_cfg.persistent_memory, dict) and agent_cfg.persistent_memory.get("vector_db_id"):
                vector_db_id = agent_cfg.persistent_memory["vector_db_id"]
        
        store_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        
        store_meta = PersistentMemoryStoreMeta(
            id=store_id,
            agent_id=agent_id,
            vector_db_id=vector_db_id,
            created_at=created_at,
            embedding_model=self.config.embedding_model,
            dimension=self.config.embedding_dimension,
            chunk_size_in_tokens=self.config.chunk_size_in_tokens,
            overlap_size_in_tokens=self.config.overlap_size_in_tokens,
            provider_id=self.config.provider_id,
        )
        
        # Persist metadata in KV store
        key = f"{self._store_prefix}{agent_id}"
        await self.kv_store.set(key, store_meta.model_dump_json())
        
        logger.info(f"Created persistent memory store for agent {agent_id} with vector_db_id {vector_db_id}")
        return store_meta
    
    async def add_turn(
        self,
        agent_id: str,
        session_id: str,
        messages: list[Message],
        output: Message,
        summarize: bool = False,
    ) -> None:
        """Add memory entries from a turn."""
        store_meta = await self.get_store_meta(agent_id)
        if not store_meta:
            logger.warning(f"No memory store found for agent {agent_id}, skipping memory storage")
            return
        
        # Collect text content to store
        texts_to_store = []
        
        # Add user messages
        for msg in messages:
            if isinstance(msg, UserMessage):
                content = self._extract_text_content(msg.content)
                if content:
                    texts_to_store.append(content)
        
        # Add output message
        if hasattr(output, "content"):
            content = self._extract_text_content(output.content)
            if content:
                texts_to_store.append(content)
        
        if not texts_to_store:
            return
        
        # Create documents to insert
        documents = []
        for text in texts_to_store:
            doc_id = str(uuid.uuid4())
            documents.append(
                RAGDocument(
                    document_id=doc_id,
                    content=text,
                    metadata={
                        "agent_id": agent_id,
                        "session_id": session_id,
                        "created_at": datetime.now(UTC).isoformat(),
                    },
                )
            )
        
        # Insert documents using VectorIO
        if documents:
            try:
                chunking_strategy = VectorStoreChunkingStrategyStatic(
                    type="static",
                    static=VectorStoreChunkingStrategyStaticConfig(
                        chunk_overlap_tokens=store_meta.overlap_size_in_tokens,
                        max_chunk_size_tokens=store_meta.chunk_size_in_tokens,
                    ),
                )
                
                await self.vector_io_api.insert(
                    vector_db_id=store_meta.vector_db_id,
                    documents=documents,
                    chunking_strategy=chunking_strategy,
                )
                logger.info(
                    f"Stored {len(documents)} memory entries for agent {agent_id} in session {session_id}"
                )
            except Exception as e:
                logger.error(f"Failed to insert memory entries: {e}")
    
    async def search(
        self, agent_id: str, query: str, max_chunks: int = 5
    ) -> list[dict]:
        """Search for relevant memory entries."""
        store_meta = await self.get_store_meta(agent_id)
        if not store_meta:
            return []
        
        try:
            result = await self.vector_io_api.query_chunks(
                vector_db_id=store_meta.vector_db_id,
                query=query,
                params={
                    "max_chunks": max_chunks,
                    "score_threshold": 0.0,
                },
            )
            return [
                {
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                }
                for chunk in result.chunks
            ]
        except Exception as e:
            logger.error(f"Failed to search memory: {e}")
            return []
    
    async def get_store_meta(
        self, agent_id: str
    ) -> Optional[PersistentMemoryStoreMeta]:
        """Get metadata for an agent's memory store."""
        key = f"{self._store_prefix}{agent_id}"
        value = await self.kv_store.get(key)
        if not value:
            return None
        
        try:
            return PersistentMemoryStoreMeta.model_validate_json(value)
        except Exception as e:
            logger.error(f"Failed to parse store metadata: {e}")
            return None
    
    def _extract_text_content(self, content) -> str:
        """Extract text content from a message."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, str):
                    texts.append(item)
                elif hasattr(item, "text"):
                    texts.append(item.text)
            return " ".join(texts)
        return ""
