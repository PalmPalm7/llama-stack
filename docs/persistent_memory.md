# Persistent Memory

Persistent memory allows agents to automatically store and retrieve conversational context across sessions without explicit client management of vector databases.

## Overview

The persistent memory service provides:
- Automatic creation of memory stores for agents
- Storage of conversational turns (user messages and assistant responses)
- Seamless integration with the RAG tool for retrieval
- Per-agent memory isolation

## Configuration

### Stack-level Configuration

Enable persistent memory globally by adding to your `StackRunConfig`:

```yaml
persistent_memory:
  enabled: true
  provider_id: faiss
  embedding_model: all-MiniLM-L6-v2
  embedding_dimension: 384
  chunk_size_in_tokens: 256
  overlap_size_in_tokens: 64
  auto_summarize_turns: false
  summary_model: null
```

For the meta-reference agents provider, add to the provider config:

```yaml
providers:
  agents:
    - provider_id: meta-reference
      provider_type: inline::meta-reference
      config:
        persistence_store:
          type: sqlite
          db_name: agents_store.db
        responses_store:
          type: sqlite
          db_name: responses_store.db
        persistent_memory:
          enabled: true
          provider_id: faiss
          embedding_model: all-MiniLM-L6-v2
          embedding_dimension: 384
```

### Agent-level Configuration

Override persistent memory settings per agent:

```python
agent_config = AgentConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    instructions="You are a helpful assistant.",
    persistent_memory={
        "enabled": True,
        "vector_db_id": "custom-memory-store"  # Optional: use existing vector DB
    }
)
```

## How It Works

### Memory Store Creation

When an agent is created with persistent memory enabled:
1. A vector database is provisioned with ID `mem-agent-{agent_id}`
2. Metadata is stored in the KV store
3. The store uses the configured embedding model and chunking parameters

### Turn Storage

After each agent turn completes:
1. User messages and assistant responses are extracted
2. Text content is chunked using the configured parameters
3. Chunks are embedded and stored in the agent's memory vector DB
4. Metadata includes agent_id, session_id, and timestamp

### Memory Retrieval

During agent turns:
1. If the RAG toolgroup is used, the agent's memory vector_db_id is automatically included
2. Relevant past conversations are retrieved based on similarity to the current query
3. Retrieved context is provided to the agent alongside any explicitly provided vector DBs

## API Usage

### Creating an Agent with Persistent Memory

```python
# Create an agent with persistent memory enabled
agent = await agents_api.create_agent(
    AgentConfig(
        model="meta-llama/Llama-3.1-8B-Instruct",
        instructions="You are a helpful assistant with persistent memory.",
        toolgroups=["builtin::rag"],  # Enable RAG for retrieval
        persistent_memory={"enabled": True}
    )
)

# Create a session
session = await agents_api.create_agent_session(
    agent_id=agent.agent_id,
    session_name="conversation-1"
)

# First turn - agent learns something
await agents_api.create_agent_turn(
    agent_id=agent.agent_id,
    session_id=session.session_id,
    messages=[
        UserMessage(content="My favorite color is blue.")
    ],
    stream=True
)

# Later, in a new session...
session2 = await agents_api.create_agent_session(
    agent_id=agent.agent_id,
    session_name="conversation-2"
)

# Agent can recall information from previous session
await agents_api.create_agent_turn(
    agent_id=agent.agent_id,
    session_id=session2.session_id,
    messages=[
        UserMessage(content="What is my favorite color?")
    ],
    stream=True
)
# Agent responds: "Your favorite color is blue."
```

## Implementation Details

### Storage

- **Metadata**: Stored in the KV store under key `pm:store:{agent_id}`
- **Memory Entries**: Stored as embeddings in the agent's vector database
- **Chunking**: Uses VectorIO with configurable chunk size and overlap

### Vector Database IDs

- Default: `mem-agent-{agent_id}`
- Custom: Can be overridden in agent configuration
- Automatically included in RAG queries when present

### Backward Compatibility

- Default behavior: Persistent memory is disabled
- Explicit opt-in required at stack or agent level
- Existing vector_db_ids continue to work unchanged
- No changes to public REST API endpoints

## Best Practices

1. **Enable RAG Toolgroup**: Include `"builtin::rag"` in agent toolgroups for automatic retrieval
2. **Tune Chunk Size**: Adjust `chunk_size_in_tokens` based on your use case (default: 256)
3. **Choose Embedding Model**: Select an appropriate model for your domain
4. **Monitor Storage**: Memory grows over time; implement retention policies if needed
5. **Test Retrieval**: Verify relevant memories are retrieved by examining RAG tool responses

## Limitations

- No automatic summarization (currently disabled)
- No built-in retention policies or quotas
- Memory is per-agent, not per-user (user-scoped support can be added later)
- Requires a vector database provider (e.g., FAISS, Chroma)

## Future Enhancements

- Automatic turn summarization
- Time-based memory retention policies
- User-scoped memory in addition to agent-scoped
- Memory compression and consolidation
- Public Memory API endpoints for explicit control
