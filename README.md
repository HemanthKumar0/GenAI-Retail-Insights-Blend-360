# Retail Insights Assistant

A GenAI-powered multi-agent system that lets you query retail sales data using natural language. Upload a CSV, ask a question in plain English, and get instant insights — no SQL required.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![DuckDB](https://img.shields.io/badge/Data-DuckDB-yellow)
![LLM](https://img.shields.io/badge/LLM-OpenAI%20%7C%20Gemini-green)

## Screenshots

> Add your screenshots to the `screenshots/` folder and reference them here.
>
> Example:
> ```
> ![Chat Interface](screenshots/chat_interface.png)
> ![Summary Mode](screenshots/summary_mode.png)
> ```

## Features

- **Natural Language Queries** — Ask questions like "What are the top 10 products by revenue?"
- **Multi-Agent Architecture** — 4 specialized agents: Query Agent, Extraction Agent, Validation Agent, Orchestrator
- **Two Modes** — Q&A Mode for specific questions, Summarization Mode for trend analysis
- **Multiple Formats** — CSV, Excel (.xlsx), JSON file support
- **Smart Caching** — LRU cache for query results + LLM response caching
- **Auto Error Recovery** — Retry logic with query reformulation on failures
- **Dual LLM Support** — OpenAI (GPT-4) and Google Gemini
- **Optional RAG** — FAISS-based semantic search for retrieval-augmented generation

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│         File Upload · Chat · Mode Select            │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────┐
│                   Orchestrator                      │
│          Coordinates · Retries · Context            │
└───┬────────────────┬────────────────┬──────────────┘
    │                │                │
    ▼                ▼                ▼
┌────────┐    ┌───────────┐    ┌────────────┐
│ Query  │    │ Extraction│    │ Validation │
│ Agent  │    │   Agent   │    │   Agent    │
│ NL→SQL │    │  DuckDB   │    │ Type/Math  │
└────┬───┘    └─────┬─────┘    └─────┬──────┘
     │              │                │
     └──────────────▼────────────────┘
              ┌─────────────┐
              │ LLM Provider│
              │ OpenAI/Gemini│
              └─────────────┘
```


## Quick Start

### 1. Clone & Install

```bash
git clone <repository-url>
cd retail-insights-assistant
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy the example env file and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
LLM_PROVIDER=openai          # or "gemini"
OPENAI_API_KEY=your_key_here
# GOOGLE_API_KEY=your_key_here  # if using Gemini
```

> ⚠️ **Never commit `.env` to git** — it's already in `.gitignore`.

### 3. Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Upload any CSV from `Sales Dataset/` and start asking questions.

## Project Structure

```
├── app.py                      # Streamlit web app
├── src/
│   ├── orchestrator.py         # Multi-agent coordinator
│   ├── query_agent.py          # NL → SQL/Pandas parser
│   ├── extraction_agent.py     # Query executor (DuckDB)
│   ├── validation_agent.py     # Result validator
│   ├── data_store.py           # Data loading & schema
│   ├── llm_provider.py         # OpenAI/Gemini abstraction
│   ├── context_manager.py      # Conversation memory
│   ├── prompt_templates.py     # LLM prompt templates
│   ├── summarization_mode.py   # Summary generation
│   ├── qa_mode.py              # Q&A processing
│   ├── rag_system.py           # FAISS semantic search
│   ├── error_handler.py        # Error management
│   ├── performance_monitor.py  # Latency & token tracking
│   ├── models.py               # Data models
│   ├── config.py               # Configuration
│   ├── schema_models.py        # Schema definitions
│   ├── llm_response_validator.py # LLM output validation
│   └── app_utils.py            # App initialization
├── tests/                      # 500+ unit & property-based tests
├── Sales Dataset/              # Sample CSV files (178K+ rows)
├── screenshots/                # UI screenshots
├── .env.example                # Environment template
├── requirements.txt            # Python dependencies
└── pytest.ini                  # Test configuration
```

## Sample Queries

**Q&A Mode:**
- "What are the total sales?"
- "Show me top 10 products by revenue"
- "Which category has the highest sales?"
- "Compare Q1 vs Q2 performance"
- "What about the lowest?" *(follow-up with context)*

**Summarization Mode:**
- "Summarize the sales data"
- "What are the key trends?"

## Configuration

All options via `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | `openai` or `gemini` | `openai` |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_MODEL` | Model name | `gpt-4-turbo-preview` |
| `GOOGLE_API_KEY` | Gemini API key | — |
| `GEMINI_MODEL` | Model name | `gemini-pro` |
| `MAX_RETRIES` | Retry attempts | `3` |
| `QUERY_TIMEOUT` | Timeout (seconds) | `30` |
| `CACHE_SIZE` | LRU cache entries | `100` |
| `ENABLE_RAG` | Enable FAISS search | `false` |

## Running Tests

```bash
pytest tests/ -v                        # Full suite
pytest tests/test_orchestrator.py -v    # Specific module
pytest tests/ --cov=src                 # With coverage
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| UI | Streamlit |
| Data Engine | DuckDB + Pandas |
| LLM | OpenAI GPT-4 / Google Gemini |
| Vector Store | FAISS (optional) |
| Testing | pytest + Hypothesis |
| Caching | LRU (query results + LLM responses) |

## Scalability Design (100GB+)

The current implementation handles demo-scale data in-memory. For production scale:

| Layer | Demo | Production |
|-------|------|-----------|
| Processing | Pandas | PySpark / Dask |
| Storage | In-memory DuckDB | BigQuery / Snowflake |
| Caching | Local LRU | Redis Cluster |
| Vector Store | FAISS | Pinecone / Weaviate |
| Deployment | Single instance | K8s cluster |

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "No API key configured" | Create `.env` with your API key |
| Module not found | `pip install -r requirements.txt` |
| Query timeout | Increase `QUERY_TIMEOUT` in `.env` |
| Out of memory | Use smaller dataset or enable pagination |

## License

MIT

---

Built for data-driven decision making.
