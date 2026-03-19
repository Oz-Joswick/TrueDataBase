-- face_system/pi/schema.sql
-- ──────────────────────────
-- Run once to set up the database.
--
-- Prerequisites:
--   1. PostgreSQL 14+ installed
--   2. pgvector extension installed (see README)
--   3. Database created:
--        sudo -u postgres createdb facedb
--
-- Run with:
--   psql -U postgres -d facedb -f schema.sql
--
-- Safe to re-run — all statements use IF NOT EXISTS / IF EXISTS guards.
-- ─────────────────────────────────────────────────────────────────────────────


-- ── Extensions ────────────────────────────────────────────────────────────────

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: nearest-neighbour search
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram index: fast name search
CREATE EXTENSION IF NOT EXISTS btree_gin;   -- GIN index support for JSONB


-- ── Tables ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS people (
    id          SERIAL          PRIMARY KEY,
    name        TEXT            NOT NULL CHECK (length(trim(name)) > 0),
    metadata    JSONB           NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

COMMENT ON TABLE  people              IS 'One row per registered individual.';
COMMENT ON COLUMN people.name         IS 'Display name. Person_XXXXXX for auto-registered unknowns.';
COMMENT ON COLUMN people.metadata     IS 'Arbitrary JSON: auto_detected, needs_review, source_job, etc.';


CREATE TABLE IF NOT EXISTS face_embeddings (
    id          SERIAL          PRIMARY KEY,
    person_id   INT             NOT NULL
                                REFERENCES people(id) ON DELETE CASCADE,
    embedding   VECTOR(512)     NOT NULL,   -- ArcFace 512-dim normalised vector
    created_at  TIMESTAMPTZ     NOT NULL DEFAULT now()
);

COMMENT ON TABLE  face_embeddings           IS 'One or more reference embeddings per person.';
COMMENT ON COLUMN face_embeddings.embedding IS '512-dim L2-normalised ArcFace embedding from InsightFace.';


-- ── Indexes ───────────────────────────────────────────────────────────────────

-- Primary vector search index.
-- ivfflat with cosine distance — good up to ~100K rows.
-- lists = sqrt(expected row count). Tune upward as DB grows:
--   1K  rows → lists = 32
--   10K rows → lists = 100
--   100K rows → lists = 316   (switch to hnsw at this scale)
CREATE INDEX IF NOT EXISTS idx_face_embeddings_ivfflat
    ON face_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Fast lookup of all embeddings for a given person (used by diversity check)
CREATE INDEX IF NOT EXISTS idx_face_embeddings_person_id
    ON face_embeddings (person_id);

-- Fast ordering by insertion time (used by get_embeddings_for_person)
CREATE INDEX IF NOT EXISTS idx_face_embeddings_created_at
    ON face_embeddings (person_id, created_at ASC);

-- JSONB GIN index — fast filtering on metadata fields e.g.
--   WHERE metadata->>'needs_review' = 'true'
CREATE INDEX IF NOT EXISTS idx_people_metadata
    ON people
    USING GIN (metadata);

-- Trigram index on name — fast partial / fuzzy name search if needed later
CREATE INDEX IF NOT EXISTS idx_people_name_trgm
    ON people
    USING GIN (name gin_trgm_ops);

-- Standard B-tree on created_at for time-range queries
CREATE INDEX IF NOT EXISTS idx_people_created_at
    ON people (created_at DESC);


-- ── Auto-update updated_at ────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_people_updated_at ON people;
CREATE TRIGGER trg_people_updated_at
    BEFORE UPDATE ON people
    FOR EACH ROW
    EXECUTE FUNCTION set_updated_at();


-- ── Useful views ──────────────────────────────────────────────────────────────

-- Summary view used by /people endpoint and dashboard viz panel
CREATE OR REPLACE VIEW people_summary AS
    SELECT
        p.id,
        p.name,
        p.metadata,
        p.created_at,
        p.updated_at,
        COUNT(fe.id)                                    AS embedding_count,
        (p.metadata->>'needs_review')::boolean          AS needs_review,
        (p.metadata->>'auto_detected')::boolean         AS auto_detected,
        MAX(fe.created_at)                              AS last_seen
    FROM people p
    LEFT JOIN face_embeddings fe ON fe.person_id = p.id
    GROUP BY p.id
    ORDER BY embedding_count DESC, p.created_at DESC;

COMMENT ON VIEW people_summary IS
    'Denormalised view of people with embedding counts and metadata flags. '
    'Used by GET /people and the dashboard.';

-- People that need manual review (auto-registered unknowns)
CREATE OR REPLACE VIEW needs_review AS
    SELECT * FROM people_summary
    WHERE needs_review IS TRUE
    ORDER BY created_at DESC;

COMMENT ON VIEW needs_review IS
    'Subset of people_summary where needs_review = true. '
    'Use this to find auto-registered Person_XXXXXX entries to rename.';


-- ── Verification query ────────────────────────────────────────────────────────
-- Run this after setup to confirm everything is in place:
--
--   SELECT schemaname, tablename
--   FROM pg_tables
--   WHERE schemaname = 'public'
--   ORDER BY tablename;
--
--   SELECT indexname, indexdef
--   FROM pg_indexes
--   WHERE tablename IN ('people', 'face_embeddings')
--   ORDER BY tablename, indexname;
--
--   SELECT * FROM people_summary LIMIT 5;


-- ── Scaling notes ─────────────────────────────────────────────────────────────
--
-- At >100K embeddings, switch from ivfflat to hnsw for better query speed:
--
--   DROP INDEX idx_face_embeddings_ivfflat;
--
--   CREATE INDEX idx_face_embeddings_hnsw
--       ON face_embeddings
--       USING hnsw (embedding vector_cosine_ops)
--       WITH (m = 16, ef_construction = 64);
--
-- hnsw uses more memory (~4GB for 100K 512-dim vectors) but query time
-- stays under 10ms regardless of table size.
-- ivfflat is preferred below 100K — lower memory, simpler to tune.
-- ─────────────────────────────────────────────────────────────────────────────