# InterVideo Pipeline – CV/JD Processing Backend

Python FastAPI backend for InterVideo. Parses CVs and job descriptions, generates interview questions via LangGraph pipeline, stores data in SQLite.

## Features

- **CV parsing** via LlamaParse (handles PDF, DOCX, images)
- **JD parsing** with structured extraction
- **LangGraph pipeline** with security, triage, enrichment, question generation, merge, judge, fitness scoring
- **FastAPI REST API** for client integration
- **SQLite storage** with full CRUD operations
- **117 passing tests** with pytest

## Tech Stack

**Backend**: Python 3.11+ + FastAPI + Uvicorn  
**Parsing**: LlamaParse (LlamaCloud API)  
**LLM**: OpenAI GPT via LangGraph + Instructor  
**Search**: Tavily API for candidate enrichment  
**Database**: SQLite  
**Testing**: pytest + pytest-asyncio

## Quick Start

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Set up API keys** in `.env`
   ```
   OPENAI_API_KEY=<your-key>
   LLAMA_CLOUD_API_KEY=<your-key>
   TAVILY_API_KEY=<your-key>
   ```

3. **Initialize database**
   ```sh
   python -m src.database.db
   ```

4. **Run dev server**
   ```sh
   uvicorn src.api.app:app --reload --port 8000
   ```

5. **Access API** at `http://localhost:8000/docs`

6. **Run Full Pipeline**
   ```sh
   uv run python run_full_pipeline.py
   ```

   To run exclusively through the HTTP API, start the API server in one terminal, then run:
   ```sh
   uv run python run_full_pipeline_with_api.py
   ```

## API

```sh
curl -F "file=@cv.pdf" http://localhost:8000/cv/upload
```

```sh
curl -F "source_type=pasted" -F "text=Senior engineer; Python and API experience required" http://localhost:8000/jd
curl -F "source_type=url" -F "url=https://example.com/jobs/role" http://localhost:8000/jd
curl -F "source_type=file" -F "file=@job-description.pdf" http://localhost:8000/jd
```

```sh
curl -X POST http://localhost:8000/pipeline/start -H "Content-Type: application/json" -d "{\"cv_id\":1,\"jd_id\":1}"
curl http://localhost:8000/pipeline/1/operations
```

## Project Structure

```
pipeline/
├── src/
│   ├── api/              # FastAPI routes
│   ├── database/         # SQLite schema + CRUD
│   ├── models/           # Pydantic schemas
│   ├── parsers/          # CV/JD parsing
│   ├── pipeline/
│   │   └── nodes/        # LangGraph nodes (question generation, merge, judge, etc.)
│   └── utils/            # Security, embeddings, tracking
├── tests/                # 117 tests
├── scripts/              # DB seed/cleanup
├── demo_*.py             # Live demos
├── outputs/              # Generated CV markdown
└── uploads/              # Uploaded CVs/JDs
```


## Testing

```bash
uv run pytest
```


## Cost

Variable based on CV/JD complexity. ~$0.05–0.20 per full pipeline run (parsing + question generation).

## Interview Control Agent

The interview control agent will manage:
- Question flow decisions (next question vs probe deeper)
- Answered questions tracking (state, timestamps)
- Exit conditions (all questions asked, time limit, fatigue)
- Answer persistence (save state after each turn)
