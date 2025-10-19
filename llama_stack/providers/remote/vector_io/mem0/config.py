# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Optional

from pydantic import BaseModel, Field


class Mem0VectorIOConfig(BaseModel):
    """Configuration for the Mem0 VectorIO provider.

    - is_cloud: Use Mem0 cloud (True) or local (False)
    - api_key: API key for cloud usage (can come from env in Mem0 client)
    - base_url: Optional cloud base URL if Mem0 supports it
    - local_config: Dict passed to Mem0 local client (required when is_cloud=False)
    - user_namespace: Prefix used to namespace user_id per vector_db_id
    - default_limit: default search limit for queries
    """

    is_cloud: bool = Field(default=True, description="Use Mem0 cloud if True; local Mem0 if False.")
    api_key: Optional[str] = Field(default=None, description="Mem0 cloud API key.")
    base_url: Optional[str] = Field(default=None, description="Mem0 cloud base URL (optional).")
    local_config: Optional[dict] = Field(default=None, description="Mem0 local config (required if is_cloud=False).")
    user_namespace: str = Field(default="llamastack", description="Prefix for Mem0 user_id namespacing.")
    default_limit: int = Field(default=5, ge=1, le=100)
