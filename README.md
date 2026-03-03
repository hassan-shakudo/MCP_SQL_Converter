# Vanna Dremio NL-to-SQL Microservice

A FastAPI-based microservice that provides natural language to SQL translation for Dremio databases using Vanna AI and OpenAI. The service exposes an OpenAI-compatible chat completions API for seamless integration.

**Features:**
- Natural language to SQL translation using GPT models
- **Semantic SQL caching** — Saves ~5-10s per similar question by caching generated SQL
- OpenAI-compatible API endpoints (streaming & non-streaming)
- ChromaDB for semantic search over schemas and SQL examples

## Overview

This service translates natural language questions into SQL queries, executes them against a Dremio data lakehouse, and returns formatted results. It uses ChromaDB for semantic search over database schemas and SQL examples, combined with OpenAI's language models for SQL generation.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │         OpenAI-Compatible API Endpoints               │  │
│  │  - /v1/chat/completions (streaming & non-streaming)   │  │
│  │  - /v1/models                                          │  │
│  │  - /query (direct testing)                             │  │
│  │  - /cache/stats, /cache/clear (cache management)       │  │
│  └───────────────────────────────────────────────────────┘  │
│                            ↓                                 │
│  ┌───────────────────────────────────────────────────────┐  │
│  │            VannaNLToSQL Engine                         │  │
│  │  - Question → SQL generation                           │  │
│  │  - Context retrieval from ChromaDB                     │  │
│  │  - Result formatting                                   │  │
│  └───────────────────────────────────────────────────────┘  │
│           ↓                              ↓                   │
│  ┌─────────────────┐          ┌─────────────────────┐       │
│  │   ChromaDB      │          │   DremioClient      │       │
│  │  Vector Store   │          │   REST API Client   │       │
│  ├─────────────────┤          └─────────────────────┘       │
│  │ - DDL Schema    │                     ↓                   │
│  │ - SQL Examples  │          ┌─────────────────────┐       │
│  │ - Documentation │          │   Dremio Database   │       │
│  │ - Query Cache   │          └─────────────────────┘       │
│  └─────────────────┘                                         │
│           ↓                                                  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              OpenAI API (GPT Models)                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Core Classes

#### 1. **DremioClient**
- Handles authentication and communication with Dremio REST API
- Methods:
  - `login()`: Authenticates and retrieves access token
  - `execute_sql(sql)`: Submits SQL jobs and polls for completion
  - Returns results as pandas DataFrames

#### 2. **VannaNLToSQL**
- Main NL-to-SQL engine combining OpenAI and ChromaDB
- Features:
  - Schema training on Dremio table structures
  - Context retrieval using semantic search
  - SQL generation with GPT models
  - **Semantic SQL caching** (L1 exact match + L3 semantic similarity)
  - Query execution and result formatting
- Methods:
  - `generate_sql(question)`: Translates natural language to SQL
  - `ask(question)`: End-to-end query processing with cache lookup
  - `get_cache_stats()`: Returns cache statistics
  - `clear_cache()`: Clears all cached queries

#### 3. **FastAPI Application**
- Provides OpenAI-compatible API endpoints
- Supports both streaming and non-streaming responses
- CORS-enabled for web client integration

### Data Flow

1. **Initialization**:
   - Connect to Dremio and authenticate
   - Initialize ChromaDB with persistent storage
   - Train on database schema and SQL examples

2. **Query Processing** (with caching):
   ```
   User Question
        ↓
   Cache Lookup (ChromaDB query_cache)
   - L1: Exact question hash match
   - L3: Semantic similarity (>95%)
        ↓
   [Cache Hit] → Use cached SQL
   [Cache Miss] → Generate SQL:
      - Context Retrieval (DDL, Examples, Docs)
      - SQL Generation (OpenAI, temp=0.1)
      - Store in cache (with deduplication)
        ↓
   SQL Execution (Dremio) — Always runs for fresh data
   - Job submission
   - Status polling
   - Result retrieval
        ↓
   Response Formatting
   - Markdown tables
   - JSON results
   - Cache hit indicator
   ```

3. **Response Types**:
   - **Streaming**: Server-sent events (SSE) with chunked responses
   - **Non-streaming**: Complete JSON response
   - **Direct**: Simple query endpoint for testing

### Database Schema

The service is configured for a ski resort budget database:

**Table**: `minio."mcp-reports-test".mcp_parquet`

**Columns**:
- `_meta_resort`: Resort name (VARCHAR)
- `_meta_proc`: Process type (VARCHAR)
- `_meta_date_start`: Start date (VARCHAR, YYYY-MM-DD)
- `_meta_date_end`: End date (VARCHAR, YYYY-MM-DD)
- `_meta_run_id`: Run ID timestamp (VARCHAR)
- `DepartmentTitle`: Department name (VARCHAR)
- `Type`: Data type (Payroll, Revenue, Visits)
- `Amount`: Dollar amount or count (DECIMAL)
- `department`: Department code (VARCHAR)
- `deptcode`: Department code number (INTEGER)

**Available Resorts**: PURGATORY, PAJARITO, SANDIA, WILLAMETTE, Sipapu, Nordic, Snowbowl

## Installation & Running

### Prerequisites

- Python 3.9+
- Access to Dremio instance
- OpenAI API key

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Dremio Configuration
DREMIO_HOST=dremio-client.hyperplane-dremio.svc.cluster.local
DREMIO_PORT=9047
DREMIO_USERNAME=admin
DREMIO_PASSWORD=your_password
DREMIO_SSL=false

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4

# ChromaDB Storage
CHROMA_PATH=/tmp/chroma_db

# SQL Cache Configuration
CACHE_ENABLED=true                    # Enable/disable SQL caching
CACHE_SIMILARITY_THRESHOLD=0.05       # Cosine distance threshold (0.05 = 95% similarity)

# Service Port
PORT=8787
```

### Quick Start

The simplest way to run the service:

```bash
./run.sh
```

This script will:
1. Install all Python dependencies from `requirements.txt`
2. Start the FastAPI server on port 8787

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

### Using Alternative Package Managers

```bash
# Using bun (preferred)
bun install
bun --bun run python main.py

# Using pnpm
pnpm install
pnpm run start
```

## API Usage

### Health Check

```bash
curl http://localhost:8787/health
```

### List Available Models

```bash
curl http://localhost:8787/v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "vanna-dremio",
      "object": "model",
      "created": 1234567890,
      "owned_by": "shakudo"
    },
    {
      "id": "vanna-nl2sql",
      "object": "model",
      "created": 1234567890,
      "owned_by": "shakudo"
    }
  ]
}
```

### Chat Completions (OpenAI-Compatible)

#### Non-Streaming Request

```bash
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vanna-dremio",
    "messages": [
      {
        "role": "user",
        "content": "What is the total budget for Purgatory resort?"
      }
    ]
  }'
```

#### Streaming Request

```bash
curl -X POST http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "vanna-dremio",
    "messages": [
      {
        "role": "user",
        "content": "Show payroll by department for Sandia"
      }
    ],
    "stream": true
  }'
```

#### Using OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed",
    base_url="http://localhost:8787/v1"
)

response = client.chat.completions.create(
    model="vanna-dremio",
    messages=[
        {"role": "user", "content": "What is the revenue for all resorts?"}
    ]
)

print(response.choices[0].message.content)
```

### Direct Query Endpoint

For testing purposes, a simplified endpoint is available:

```bash
curl -X POST http://localhost:8787/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How many visits for each resort?"
  }'
```

Response:
```json
{
  "question": "How many visits for each resort?",
  "sql": "SELECT _meta_resort, SUM(Amount) as total_visits FROM minio.\"mcp-reports-test\".mcp_parquet WHERE Type = 'Visits' GROUP BY _meta_resort",
  "results": [
    {"_meta_resort": "PURGATORY", "total_visits": 15234.0},
    {"_meta_resort": "SANDIA", "total_visits": 8976.0}
  ],
  "result_text": "| _meta_resort | total_visits |\n|:-------------|-------------:|\n| PURGATORY    | 15234.0      |\n| SANDIA       | 8976.0       |",
  "row_count": 2,
  "cache_hit": null,
  "timing": {
    "total_seconds": 25.32,
    "cache_hit_type": "miss"
  }
}
```

### Cache Management Endpoints

#### Get Cache Statistics

```bash
curl http://localhost:8787/cache/stats
```

Response:
```json
{
  "enabled": true,
  "entry_count": 42,
  "similarity_threshold": 0.05,
  "chroma_path": "/tmp/chroma_db"
}
```

#### Clear Cache

```bash
curl -X DELETE http://localhost:8787/cache/clear
```

Response:
```json
{
  "success": true,
  "message": "Cache cleared"
}
```

## Example Questions

The service comes pre-trained with example queries:

- "What is the total budget for Purgatory resort?"
- "Show payroll by department for Sandia"
- "What is the revenue for all resorts?"
- "How many visits for each resort?"
- "Show budget for the last 7 days"

## Dependencies

- **FastAPI**: Web framework for building APIs
- **Uvicorn**: ASGI server for running FastAPI
- **OpenAI**: GPT model integration for SQL generation
- **ChromaDB**: Vector database for semantic search
- **Pandas**: Data manipulation and result formatting
- **httpx**: Async HTTP client for Dremio API
- **Pydantic**: Data validation and settings management
- **PyArrow**: Apache Arrow data format support
- **SQLParse**: SQL query parsing and formatting

## Configuration

### Customizing the Table Schema

To adapt this service for a different database table, modify the `_train_on_schema()` method in the `VannaNLToSQL` class:

1. Update the DDL documentation
2. Add relevant SQL examples
3. Update documentation strings
4. Modify `TABLE_NAME` constant

### Adjusting SQL Generation

The SQL generation behavior can be customized in the `generate_sql()` method:

- **Temperature**: Currently set to 0.1 for deterministic output
- **System Prompt**: Defines rules and constraints for SQL generation
- **Context Retrieval**: Adjust number of examples retrieved from ChromaDB

## Deployment

### Shakudo Platform

This microservice is designed to run on the Shakudo platform:

1. Ensure the git repository is synced
2. Deploy as a Shakudo microservice
3. The service will be accessible at `https://vanna-dremio-nl2sql.dev.hyperplane.dev`

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .

EXPOSE 8787
CMD ["python", "main.py"]
```

### Kubernetes

The service expects Dremio to be available at:
```
dremio-client.hyperplane-dremio.svc.cluster.local:9047
```

## Monitoring & Debugging

### Logs

The service uses Python's logging module. Set log level via:

```python
logging.basicConfig(level=logging.DEBUG)
```

### Key Log Messages

- "Successfully authenticated with Dremio" - Dremio connection established
- "Schema training complete" - ChromaDB initialized
- "SQL Cache initialized" - Cache ready with entry count
- "Generated SQL: [query]" - SQL generated for user question
- "Error executing SQL: [error]" - Query execution failed

### Cache Log Messages

- `[CACHE HIT - EXACT]` - Exact question match found
- `[CACHE HIT - SEMANTIC]` - Similar question found (shows similarity %)
- `[CACHE MISS]` - No cached SQL found, generating new
- `[CACHE STORE]` - New SQL cached
- `[CACHE DEDUP]` - SQL already cached from different question (skipped)
- `[QUERY COMPLETE]` - Summary with timing and cache status

### Common Issues

1. **Dremio Connection Failed**
   - Check `DREMIO_HOST`, `DREMIO_PORT`, and credentials
   - Verify network connectivity to Dremio instance

2. **OpenAI API Errors**
   - Verify `OPENAI_API_KEY` is valid
   - Check model name in `OPENAI_MODEL`

3. **ChromaDB Errors**
   - Ensure `CHROMA_PATH` directory is writable
   - Check disk space for vector storage

## Contributing

When making changes:

1. Follow the code style guidelines in `/CLAUDE.md`
2. Test changes with the direct query endpoint first
3. Verify OpenAI-compatible API compatibility
4. Update documentation for new features

## License

Internal Shakudo project.
