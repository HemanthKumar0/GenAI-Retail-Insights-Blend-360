# Retail Insights Assistant — Complete Project Walkthrough

## The Big Picture

User opens a Streamlit web app → uploads a CSV → types a question like "What are the top 5 products by revenue?" → the system converts that English into SQL using an LLM → runs it on DuckDB → validates the results → uses the LLM again to write a nice English answer → shows it in the chat.

---

## File-by-File Breakdown

### `app.py` — The Streamlit UI (entry point)

This is what the user sees. You run `streamlit run app.py`.

- `initialize_session_state()` — creates the orchestrator once and stores it in Streamlit's session. Also initializes chat history, mode, loaded datasets. **No LLM used.**
- `display_header()` — renders the title and subtitle. Pure UI.
- `display_sidebar()` — shows mode toggle (Q&A vs Summarization), file uploader, loaded datasets list, API key status. When a file is uploaded, it calls `data_store.load_csv()` / `load_excel()` / `load_json()` to load it into DuckDB. **No LLM used.**
- `display_chat_history()` — loops through `st.session_state.messages` and renders each one as a chat bubble with metadata (time, tokens, cached). Pure UI.
- `process_user_input(user_input)` — the core action. Takes the user's typed question, calls `orchestrator.process_query(user_input, mode)`, displays the response, and appends both user + assistant messages to history. **This is where the LLM pipeline kicks off.**
- `display_example_queries()` — shows clickable example buttons. Pure UI.
- `main()` — ties everything together. Calls all the above in order.

---

### `src/utils/app_utils.py` — App Initialization

Single function:
- `create_orchestrator()` — reads `Config` to decide OpenAI vs Gemini, creates the LLM provider, creates `DataStore`, creates all 3 agents, wires them into the `Orchestrator`. **No LLM calls here**, just object construction.

---

### `src/core/config.py` — Configuration

- `Config` class — reads `.env` file via `python-dotenv`. Holds all settings: API keys, model names, timeouts, cache size, RAG toggle, etc. **No LLM used.**
- `setup_logging()` — configures Python logging to console + `retail_insights.log` file. Called on import.

---

### `src/core/models.py` — Data Models (the "language" of the system)

All plain Python dataclasses. **No LLM used anywhere.** These are just data containers passed between agents:

- `Message` — a chat message (role, content, timestamp, metadata)
- `StructuredQuery` — what the QueryAgent produces: `operation_type` (sql/pandas/semantic), `operation` (the actual SQL string), `explanation`, `parameters`
- `QueryResult` — what the ExtractionAgent produces: a DataFrame + row_count + execution_time + cached flag
- `Anomaly` — a data quality issue (type, description, severity, affected_rows)
- `ValidationResult` — what the ValidationAgent produces: passed/failed + issues list + anomalies + confidence score
- `Response` — the final answer: natural language text + optional DataFrame + metadata
- `ColumnInfo`, `TableSchema`, `DataSchema` — schema descriptions of loaded tables

---

### `src/core/orchestrator.py` — The Brain (LangGraph)

This is the most important file. It coordinates everything using a LangGraph `StateGraph`.

**The graph looks like this:**
```
parse_query → execute_query → validate_results → [conditional]
                                                    ├─ PASSED → format_response → END
                                                    ├─ FAILED (retries left) → reformulate_query → parse_query (loop back)
                                                    └─ FAILED (max retries) → error_response → END
```

**`AgentState`** — a TypedDict that flows through every node. Contains: user_query, current_query, schema, structured_query, query_result, validation_result, attempt count, response, error.

**Key functions:**

- `__init__()` — stores all agents, builds the LangGraph once via `_build_graph()`. **No LLM call.**
- `_build_graph()` — registers 6 nodes and their edges, compiles the graph. **No LLM call.**
- `process_query(user_query, mode)` — the public entry point. Adds user message to context, builds initial state dict, calls `self._graph.invoke(initial_state)`, adds assistant response to context. **This triggers the entire LLM pipeline.**
- `_node_parse_query(state)` — calls `query_agent.parse_query()`. **🔴 USES LLM** (inside QueryAgent)
- `_node_execute_query(state)` — calls `extraction_agent.execute_query()`. **No LLM.** Pure DuckDB/Pandas.
- `_node_validate_results(state)` — calls `validation_agent.validate_results()`. **No LLM.** Pure math/logic checks.
- `_route_after_validation(state)` — decides: format_response vs reformulate vs error. **No LLM.** Just if/else logic.
- `_node_format_response(state)` — calls `_format_response()` which uses LLM to write the English answer. **🔴 USES LLM**
- `_node_reformulate_query(state)` — calls `_reformulate_query()` which asks LLM to rewrite the query. **🔴 USES LLM**
- `_node_error_response(state)` — builds a static error message. **No LLM.**
- `_get_data_schema()` — reads table schemas from DataStore. **No LLM.**
- `_log_communication()` — appends to communication_log list. **No LLM.**
- `_create_data_summary()` — converts DataFrame to text for the LLM prompt. **No LLM** (just string formatting).
- `reset_context()`, `get_conversation_history()`, `get_communication_log()`, `clear_communication_log()` — simple getters/setters. **No LLM.**

---

### `src/agents/query_agent.py` — Agent 1: Natural Language → SQL

This is where English becomes SQL. **🔴 Heavy LLM usage.**

- `__init__(llm_provider)` — stores the LLM provider.
- `parse_query(query, schema, context)` — the main function. Formats a prompt with the schema + conversation context + user query, sends it to the LLM, gets back JSON like `{"operation_type": "sql", "operation": "SELECT ...", "explanation": "..."}`, parses it into a `StructuredQuery`. **🔴 USES LLM**
- `_parse_llm_response(response_text)` — extracts JSON from LLM output (handles markdown code blocks), validates required fields, creates `StructuredQuery`. **No LLM** (just parsing).
- `_extract_json(text)` — regex to pull JSON from markdown blocks. **No LLM.**
- `_schema_to_dict(schema)` — converts DataSchema to dict for the prompt. **No LLM.**

---

### `src/agents/extraction_agent.py` — Agent 2: Execute the Query

This agent runs the actual SQL/Pandas against your data. **No LLM used at all.**

- `__init__(data_store)` — stores DataStore reference, sets up LRU cache.
- `execute_query(query: StructuredQuery)` — the main function. Checks cache first, then routes to `_execute_sql_query()` or `_execute_pandas_query()` based on `operation_type`. Applies timeout check and pagination (caps at 10,000 rows). Caches the result.
- `_execute_sql_query(query)` — calls `data_store.execute_sql(query.operation)`. DuckDB runs the SQL.
- `_execute_pandas_query(query)` — uses `eval()` with restricted namespace to run Pandas expressions on the DataFrame.
- `_get_cache_key(query)` — MD5 hash of the query for caching.
- `_get_from_cache()` / `_add_to_cache()` — LRU cache management.
- `clear_cache()` — empties the cache.

---

### `src/agents/validation_agent.py` — Agent 3: Check the Results

Validates data quality. **No LLM used at all.** Pure math and logic.

- `__init__(business_rules)` — loads default or custom business rules.
- `validate_results(results, query)` — runs 5 checks in sequence, collects issues and anomalies, calculates confidence score, returns `ValidationResult(passed=True/False)`.
- `_validate_data_types(df)` — checks for object columns that should be numeric, datetime columns with NaT values.
- `_check_mathematical_consistency(df)` — checks: negative values in sales columns, total ≠ subtotal + tax, amount ≠ price × quantity.
- `_validate_empty_results(results, query)` — if 0 rows returned, checks if WHERE/JOIN/HAVING might be too restrictive.
- `_detect_anomalies(df)` — finds negative sales, invalid dates (NaT, out-of-range 1900-2100).
- `_calculate_confidence(issues, anomalies)` — starts at 1.0, subtracts 0.1 per issue, 0.2 per error anomaly, 0.05 per warning.
- `check_business_rules(results)` — validates category values against known list, checks sales/order count consistency, checks sales within min/max bounds.

---

### `src/data/data_store.py` — Data Loading & SQL Engine

Manages DuckDB in-memory database. **No LLM used.**

- `__init__()` — creates DuckDB in-memory connection.
- `register_dataframe(table_name, df)` — registers a Pandas DataFrame as a DuckDB table.
- `execute_sql(query)` — runs SQL on DuckDB, auto-converts backticks to double quotes, returns DataFrame.
- `get_table_schema(table_name)` — returns column names, types, nullable flags, row count.
- `list_tables()` — returns list of registered table names.
- `load_csv(file_path)` — loads CSV. If >1GB, uses `_load_csv_chunked()` which reads in 100K-row chunks and appends to DuckDB table.
- `load_excel(file_path)` — loads each sheet as a separate table.
- `load_json(file_path)` — loads JSON with `_normalize_json()` to flatten nested structures.
- `close()` — closes DuckDB connection.

---

### `src/data/context_manager.py` — Conversation Memory

Manages chat history within token limits. **🔴 Uses LLM only for summarization.**

- `ContextWindow` — low-level storage. `add_message()`, `get_context_string()` (truncates to fit token budget, prioritizes recent messages), `estimate_tokens()`.
- `ContextManager` — high-level wrapper. `add_message()` checks if context is >80% full and triggers `_summarize_old_context()`. `get_context()` returns the context string.
- `_summarize_old_context()` — keeps last 3 messages, sends older ones to LLM with a summarization prompt, stores the summary. **🔴 USES LLM**

---

### `src/llm/llm_provider.py` — LLM Abstraction Layer

- `LLMResponse` — dataclass: content, tokens_used, model, cached flag.
- `LLMProvider` (abstract base) — defines `generate()`, `count_tokens()`, `generate_with_cache()`, `generate_with_retry()`.
- `generate_with_cache()` — SHA256 hashes the prompt+params, checks in-memory dict cache, returns cached response or generates new one.
- `generate_with_retry()` — exponential backoff (1s, 2s, 4s) on failures.
- `GeminiProvider` — implements `generate()` using `google.generativeai`, `count_tokens()` using Gemini's tokenizer.
- `OpenAIProvider` — implements `generate()` using `openai.ChatCompletion`, `count_tokens()` using `tiktoken`.
- `LLMProviderFactory` — `create_provider("gemini"/"openai", model, api_key)`.

---

### `src/llm/prompt_templates.py` — All LLM Prompts

Contains every prompt template used in the system. **No LLM calls here** — just string templates.

- `QUERY_PARSING_PROMPT` — the big one. Includes schema, context, user query, 7 few-shot examples showing how to convert English → SQL/Pandas/semantic. Includes DuckDB-specific rules (double quotes, TRY_CAST).
- `SUMMARIZATION_PROMPT` — tells LLM to write a business summary with sections (Overview, Key Metrics, Top Performers, Concerns, Recommendations).
- `CONTEXT_SUMMARIZATION_PROMPT` — summarizes old conversation history.
- `format_*()` static methods — fill in the templates with actual data.

---

### `src/modes/qa_mode.py` — Q&A Mode

- `answer_question(question)` — checks if it's a clarification request, otherwise calls `orchestrator.process_query(question, mode="qa")`. **🔴 USES LLM** (via orchestrator)
- `_is_clarification_request(question)` — keyword matching for "explain", "clarify", "tell me more", etc. **No LLM.**
- `_handle_clarification(question)` — gets last assistant response from history, asks LLM to elaborate. **🔴 USES LLM**

---

### `src/modes/summarization_mode.py` — Summarization Mode

- `generate_summary(table_name)` — the main function. Calls 4 analysis methods, then formats with LLM.
- `_get_dataset_info()` — basic stats (row count, columns, date range). **No LLM.**
- `_calculate_key_metrics()` — sums, averages, min/max for numeric columns. Calculates YoY growth. **No LLM.**
- `_calculate_yoy_growth()` — groups by year, computes percentage change. **No LLM.**
- `_identify_performers()` — groups by first categorical column, sorts by first numeric column, gets top 5 and bottom 5. **No LLM.**
- `_detect_trends()` — IQR outlier detection, rolling average trend direction (increasing/decreasing/stable). **No LLM.**
- `_format_summary()` — sends all the above data to LLM with the summarization prompt template. **🔴 USES LLM**
- `_generate_fallback_summary()` — if LLM fails, builds a basic text summary. **No LLM.**

---

## The Complete Flow (step by step)

Here's what happens when you type "What are the top 5 products by revenue?":

```
1. app.py: process_user_input() receives the text
2. app.py: calls orchestrator.process_query("What are the top 5...", mode="qa")
3. orchestrator.py: adds user message to ContextManager
4. orchestrator.py: reads table schema from DataStore (column names, types)
5. orchestrator.py: gets conversation context string from ContextManager
6. orchestrator.py: builds initial AgentState dict, calls self._graph.invoke()

   --- LangGraph starts ---

7. NODE parse_query:
   → QueryAgent.parse_query() is called
   → Builds a prompt with schema + context + user question + few-shot examples
   → 🔴 Sends prompt to LLM (OpenAI/Gemini)
   → LLM returns JSON: {"operation_type": "sql", "operation": "SELECT product,
     SUM(sales) as revenue FROM sales GROUP BY product ORDER BY revenue DESC
     LIMIT 5", "explanation": "Top 5 products by revenue"}
   → Parses JSON into StructuredQuery object

8. NODE execute_query:
   → ExtractionAgent.execute_query() is called
   → Checks LRU cache (miss on first run)
   → Runs SQL on DuckDB: SELECT product, SUM(sales)...
   → Gets back a Pandas DataFrame with 5 rows
   → Wraps in QueryResult (data, row_count=5, execution_time=0.02s)
   → Stores in cache

9. NODE validate_results:
   → ValidationAgent.validate_results() is called
   → Check 1: data types OK
   → Check 2: no negative values in sales column
   → Check 3: not empty (5 rows)
   → Check 4: no anomalies
   → Check 5: business rules OK
   → Returns ValidationResult(passed=True, confidence=1.0)

10. ROUTING _route_after_validation:
    → validation.passed is True → go to "format_response"

11. NODE format_response:
    → Builds a text summary of the 5-row DataFrame
    → 🔴 Sends to LLM with response formatting prompt
    → LLM writes: "The top 5 products by revenue are:
      1. Product X ($50,000)..."
    → Wraps in Response object with metadata

    --- LangGraph ends ---

12. orchestrator.py: adds assistant message to ContextManager
13. app.py: displays the answer in chat bubble
14. app.py: shows the DataFrame as a table below the answer
15. app.py: shows metadata (execution time, tokens used)
```

**If validation FAILS** (say the SQL returned negative sales values):
- Step 10 routes to `reformulate_query` instead
- **🔴 LLM** rewrites the query to fix the issue
- Loops back to step 7 with the new query
- Tries up to 3 times total
- If all 3 fail → `error_response` node builds a static error message

---

## Summary: Where LLM is Used vs Not

| Step | LLM? | What does it |
|---|---|---|
| File upload & loading | ❌ | Pandas/DuckDB reads CSV/Excel/JSON |
| Schema detection | ❌ | Pandas dtype inspection |
| English → SQL conversion | ✅ | QueryAgent sends prompt to LLM |
| SQL execution | ❌ | DuckDB runs the SQL |
| Result validation | ❌ | Math checks, type checks, business rules |
| Query reformulation (on retry) | ✅ | LLM rewrites the failed query |
| Response formatting | ✅ | LLM writes the English answer |
| Summarization analysis | ❌ | Pandas groupby, aggregation, IQR |
| Summary formatting | ✅ | LLM writes the summary text |
| Context summarization | ✅ | LLM summarizes old chat history |
| Error messages | ❌ | Hardcoded templates |
| Caching | ❌ | SHA256 hash + in-memory dict |
