# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Optional

from llama_stack.apis.common.content_types import InterleavedContent
from llama_stack.apis.files import Files
from llama_stack.apis.inference import Inference
from llama_stack.apis.vector_dbs import VectorDB
from llama_stack.apis.vector_io import (
    Chunk,
    ChunkMetadata,
    QueryChunksResponse,
    VectorIO,
)
from llama_stack.log import get_logger
from llama_stack.providers.datatypes import VectorDBsProtocolPrivate
from llama_stack.providers.utils.inference.prompt_adapter import interleaved_content_as_str
from llama_stack.providers.utils.vector_io.vector_utils import generate_chunk_id

from .config import Mem0VectorIOConfig

log = get_logger(name=__name__, category="vector_io::mem0")
logger = logging.getLogger(__name__)

try:
    # Cloud client
    from mem0 import MemoryClient as Mem0CloudClient  # type: ignore
    # Local client
    from mem0 import Memory as Mem0Local  # type: ignore
except Exception:  # pragma: no cover
    Mem0CloudClient = None
    Mem0Local = None


def _ns_user_id(config: Mem0VectorIOConfig, vector_db_id: str) -> str:
    return f"{config.user_namespace}:{vector_db_id}"


class Mem0VectorIOImpl(VectorIO, VectorDBsProtocolPrivate):
    """VectorIO adapter backed by Mem0 (cloud/local).

    Maps vector_db_id to a Mem0 user_id namespace to isolate memory per DB.
    """

    def __init__(
        self,
        config: Mem0VectorIOConfig,
        inference_api: Optional[Inference] = None,
        files_api: Optional[Files] = None,
    ) -> None:
        self.config = config
        self.inference_api = inference_api
        self.files_api = files_api
        self._client: Any = None
        self._vector_dbs: dict[str, VectorDB] = {}

    async def initialize(self) -> None:
        if self.config.is_cloud:
            assert Mem0CloudClient, "mem0 package not available"
            kwargs = {}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = Mem0CloudClient(api_key=self.config.api_key, **kwargs)  # type: ignore
        else:
            assert Mem0Local, "mem0 local package not available"
            assert self.config.local_config, "local_config is required when is_cloud=False"
            self._client = Mem0Local.from_config(config_dict=self.config.local_config)  # type: ignore

    async def shutdown(self) -> None:
        # Mem0 client does not require explicit shutdown
        pass

    async def list_vector_dbs(self) -> list[VectorDB]:
        return list(self._vector_dbs.values())

    async def register_vector_db(
        self,
        vector_db: VectorDB,
    ) -> None:
        # Embedding fields are recorded for parity; Mem0 manages its own embedding internally
        self._vector_dbs[vector_db.identifier] = vector_db

    async def unregister_vector_db(self, vector_db_id: str) -> None:
        user_id = _ns_user_id(self.config, vector_db_id)
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self._client.delete_all(user_id=user_id)  # type: ignore
        except Exception as e:
            logger.warning(f"Mem0 delete_all failed for {user_id}: {e}")
        self._vector_dbs.pop(vector_db_id, None)

    async def insert_chunks(
        self,
        vector_db_id: str,
        chunks: list[Chunk],
        ttl_seconds: int | None = None,
    ) -> None:
        if not chunks:
            return
        user_id = _ns_user_id(self.config, vector_db_id)

        for chunk in chunks:
            content = chunk.content
            cm = getattr(chunk, "chunk_metadata", None)
            chunk_id = getattr(cm, "chunk_id", None) if cm else None
            document_id = getattr(cm, "document_id", "unknown") if cm else "unknown"
            if not chunk_id:
                chunk_id = generate_chunk_id(document_id, str(content))
            base_meta = {
                "vector_db_id": vector_db_id,
                "chunk_id": chunk_id,
                "document_id": document_id,
                **(chunk.metadata or {}),
            }
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    kwargs = {} if self._client.__class__.__name__ == "Memory" else {"output_format": "v1.1"}
                    self._client.add(
                        [{"role": "user", "content": str(content)}],
                        user_id=user_id,
                        metadata=base_meta,
                        **kwargs,
                    )  # type: ignore
            except Exception as e:
                logger.error(f"Error adding chunk to Mem0 ({vector_db_id}): {e}")
                raise

    async def query_chunks(
        self,
        vector_db_id: str,
        query: InterleavedContent,
        params: dict[str, Any] | None = None,
    ) -> QueryChunksResponse:
        text_query = interleaved_content_as_str(query)
        params = params or {}
        k = params.get("max_chunks", self.config.default_limit)
        
        user_id = _ns_user_id(self.config, vector_db_id)
        aggregated: list[Chunk] = []
        scores: list[float] = []

        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                results = self._client.search(text_query, user_id=user_id, limit=k)  # type: ignore
        except Exception as e:
            logger.error(f"Error querying Mem0 for {vector_db_id}: {e}")
            return QueryChunksResponse(chunks=[], scores=[])

        items = results.get("results", results) if isinstance(results, dict) else results
        for r in items or []:
            text = r.get("memory", "")
            meta = r.get("metadata", {}) or {}
            score = float(r.get("score", 0.0))
            chunk_id = meta.get("chunk_id") or generate_chunk_id(meta.get("document_id", "mem0_doc"), text)
            document_id = meta.get("document_id", "mem0_doc")
            chunk = Chunk(
                content=text,
                metadata={**meta, "score": score, "vector_db_id": vector_db_id},
                chunk_metadata=ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source=meta.get("source", "mem0"),
                ),
            )
            aggregated.append(chunk)
            scores.append(score)

        return QueryChunksResponse(chunks=aggregated, scores=scores)

    async def delete_chunks(self, vector_db_id: str, chunk_ids: list[str]) -> None:
        # Not implemented yet; Mem0 API may not support per-chunk deletes reliably
        raise NotImplementedError("Per-chunk delete is not supported by the Mem0 adapter yet.")
