-- Intervideo-Pipeline Database Schema

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
    run_id INTEGER NOT NULL REFERENCES pipeline_runs(id),
    source_type TEXT NOT NULL CHECK(source_type IN ('project','skill')),
    source_ref TEXT,
    question_text TEXT NOT NULL,
    question_hash TEXT NOT NULL UNIQUE,
    evidence_snippet TEXT,
    evidence_span JSON,
    level TEXT CHECK(level IN ('junior','mid','senior')),
    difficulty INTEGER CHECK(difficulty BETWEEN 1 AND 5),
    skill TEXT,
    question_type TEXT CHECK(question_type IN ('behavioral','technical','design','situational')),
    key_points JSON,
    follow_ups JSON,
    gen_model TEXT,
    prompt_version TEXT,
    judge_version TEXT,
    judge_clarity_score REAL CHECK(judge_clarity_score BETWEEN 0 AND 1),
    judge_relevance_score REAL CHECK(judge_relevance_score BETWEEN 0 AND 1),
    judge_difficulty_match_score REAL CHECK(judge_difficulty_match_score BETWEEN 0 AND 1),
    judge_groundedness_score REAL CHECK(judge_groundedness_score BETWEEN 0 AND 1),
    judge_redundancy_score REAL CHECK(judge_redundancy_score BETWEEN 0 AND 1),
    judge_overall_score REAL,
    judge_pass INTEGER,
    judge_feedback JSON,
    status TEXT NOT NULL DEFAULT 'candidate'
        CHECK(status IN ('rejected_auto','candidate','verified','rejected_human')),
    review_note TEXT,
    langsmith_run_id TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_q_status ON questions(status, skill, level);
CREATE INDEX IF NOT EXISTS idx_q_run ON questions(run_id);

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

-- Interview Sessions (Phase 11b)
CREATE TABLE IF NOT EXISTS interview_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    run_id INTEGER NOT NULL,
    total_questions INTEGER NOT NULL,
    planned_duration_seconds INTEGER NOT NULL,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    total_duration_seconds INTEGER,
    avg_answer_quality REAL,
    fatigue_detected INTEGER DEFAULT 0,
    exit_reason TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES pipeline_runs(id)
);

-- Interview Answers (Phase 11b)
CREATE TABLE IF NOT EXISTS interview_answers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    question_id INTEGER NOT NULL,
    question_text TEXT NOT NULL,
    question_level TEXT CHECK(question_level IN ('junior', 'mid', 'senior')),
    question_type TEXT CHECK(question_type IN ('behavioral', 'technical', 'design', 'situational')),
    asked_at TIMESTAMP,
    answer_text TEXT NOT NULL,
    answered_at TIMESTAMP,
    duration_seconds INTEGER,
    quality TEXT CHECK(quality IN ('excellent', 'good', 'fair', 'poor', 'off_topic')),
    feedback TEXT,
    follow_ups_asked INTEGER DEFAULT 0,
    follow_up_texts JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES interview_sessions(session_id),
    FOREIGN KEY (question_id) REFERENCES questions(id)
);

-- Interview Decisions (Phase 11b)
CREATE TABLE IF NOT EXISTS interview_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    answer_id INTEGER NOT NULL,
    action TEXT CHECK(action IN ('continue_normal', 'probe_deeper', 'skip_question', 'time_limit', 'fatigue', 'all_asked', 'candidate_exit')),
    follow_up TEXT,
    reasoning TEXT,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES interview_sessions(session_id),
    FOREIGN KEY (answer_id) REFERENCES interview_answers(id)
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
CREATE INDEX IF NOT EXISTS idx_interview_session ON interview_sessions(run_id);
CREATE INDEX IF NOT EXISTS idx_interview_answers ON interview_answers(session_id);
CREATE INDEX IF NOT EXISTS idx_interview_decisions ON interview_decisions(session_id);
