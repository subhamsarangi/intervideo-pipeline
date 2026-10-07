# InterVideo Pipeline – CV/JD Processing Backend

Python FastAPI backend for InterVideo. Parses CVs and job descriptions, generates interview questions via LangGraph pipeline, stores data in SQLite.

## Features

- **CV parsing** via LlamaParse (handles PDF, DOCX, images)
- **JD parsing** with structured extraction
- **LangGraph pipeline** with security, triage, enrichment, question generation, merge, judge, fitness scoring
- **FastAPI REST API** for client integration
- **SQLite storage** with full CRUD operations
- **97 passing tests** with pytest

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
   pip install -e .
   ```

2. **Set up API keys** in `.env`
   ```
   OPENAI_API_KEY=<your-key>
   LLAMA_CLOUD_API_KEY=<your-key>
   TAVILY_API_KEY=<your-key>
   ```

3. **Initialize database**
   ```bash
   python -m src.database.migrate
   ```

4. **Run dev server**
   ```bash
   uvicorn src.api.app:app --reload
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
│   ├── pipeline/         # LangGraph nodes + graph
│   └── utils/            # Security, embeddings, tracking
├── tests/                # 97 tests
├── scripts/              # DB seed/cleanup
├── outputs/              # Generated CV markdown
└── uploads/              # Uploaded CVs/JDs
```

## Testing

```bash
pytest
```

## Cost

Variable based on CV/JD complexity. ~$0.05–0.20 per full pipeline run (parsing + question generation).

## Requirements

- Python 3.11+
- API keys for OpenAI, LlamaCloud, Tavily
- ~10MB disk per processed CV

---

**Part of InterVideo monorepo. See `/meta/brief.md` for full system architecture.**
