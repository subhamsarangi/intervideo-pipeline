-- RecRoot Database Schema

-- CV Documents
CREATE TABLE IF NOT EXISTS cv_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    parsed_data JSON,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    llm_calls_count INTEGER DEFAULT 0,
    llm_calls_log JSON,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- JD Documents
CREATE TABLE IF NOT EXISTS jd_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    parsed_data JSON,
    source_type TEXT CHECK(source_type IN ('pasted', 'url', 'file')),
    source_url TEXT,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    error_message TEXT,
    llm_calls_count INTEGER DEFAULT 0,
    llm_calls_log JSON,
    total_input_tokens INTEGER DEFAULT 0,
    total_output_tokens INTEGER DEFAULT 0,
    total_cost_usd REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline Runs
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cv_id INTEGER NOT NULL,
    jd_id INTEGER NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed')),
    prompt_version TEXT,
    output_json JSON,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (cv_id) REFERENCES cv_documents(id),
    FOREIGN KEY (jd_id) REFERENCES jd_documents(id)
);

-- Pipeline Steps (execution log)
CREATE TABLE IF NOT EXISTS pipeline_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    step_name TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'running', 'completed', 'failed')),
    output_json JSON,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

-- Generated Questions
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    source_type TEXT CHECK(source_type IN ('project', 'skill')),
    source_id TEXT,
    question_text TEXT NOT NULL,
    evidence_snippet TEXT,
    evidence_span JSON,
    judge_relevance_score REAL,
    judge_groundedness_score REAL,
    judge_redundancy_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

-- Fitness Scores
CREATE TABLE IF NOT EXISTS fitness_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    requirement_text TEXT NOT NULL,
    requirement_type TEXT CHECK(requirement_type IN ('must_have', 'nice_to_have')),
    match_score REAL,
    evidence TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

-- Run Summaries
CREATE TABLE IF NOT EXISTS run_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL UNIQUE,
    total_questions INTEGER,
    avg_judge_score REAL,
    fitness_score REAL,
    fitness_weighted_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

-- LangGraph Checkpoints (handled by SqliteSaver, but keep schema for reference)
CREATE TABLE IF NOT EXISTS checkpoint (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_values BLOB NOT NULL,
    checkpoint_metadata BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE TABLE IF NOT EXISTS writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    channel TEXT NOT NULL,
    version INTEGER NOT NULL,
    data_value BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_cv_status ON cv_documents(status);
CREATE INDEX IF NOT EXISTS idx_jd_status ON jd_documents(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_cv_jd ON pipeline_runs(cv_id, jd_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_steps_run ON pipeline_steps(run_id);
CREATE INDEX IF NOT EXISTS idx_steps_status ON pipeline_steps(status);
CREATE INDEX IF NOT EXISTS idx_questions_run ON questions(run_id);
CREATE INDEX IF NOT EXISTS idx_fitness_run ON fitness_scores(run_id);
