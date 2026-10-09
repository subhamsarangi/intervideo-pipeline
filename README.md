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
**LLM**: OpenAI GPT-4o via LangGraph + Instructor  
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
   ```bash
   python -m src.database.db
   ```

4. **Run dev server**
   ```bash
   uvicorn src.api.app:app --reload --port 8000
   ```

5. **Access API** at `http://localhost:8000/docs`

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

## Pipeline Phases (Implemented)

- **Phase 7:** Project + Skill questions (3-5 per project, 2-3 per skill)
- **Phase 8:** Merge/dedupe/rerank (embeddings-based dedup, coverage check, JD relevance rerank)
- **Phase 9:** LLM judge evaluation (in progress)
- **Phase 10:** CV-JD fitness scoring

## Testing

```bash
uv run pytest
```

All tests pass. Coverage: parsers, security, triage, enrichment, generation, merge, dedup.

## Cost

Variable based on CV/JD complexity. ~$0.05–0.20 per full pipeline run (parsing + question generation).

## Interview Control Agent (Phase 11b, Coming)

The interview control agent will manage:
- Question flow decisions (next question vs probe deeper)
- Answered questions tracking (state, timestamps)
- Exit conditions (all questions asked, time limit, fatigue)
- Answer persistence (save state after each turn)

---

**Part of InterVideo monorepo. See `/meta/brief.md` for full system architecture.**
