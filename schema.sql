-- ============================================================================
-- Civic Infrastructure Incident Database Schema
-- Target: PostgreSQL 16+
-- Extensions: PostGIS, pgvector, uuid-ossp
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Enable Required Extensions
-- ----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ----------------------------------------------------------------------------
-- 2. Custom Enumerated Types
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'incident_source') THEN
        CREATE TYPE incident_source AS ENUM (
            'citizen_upload',
            'dashcam',
            'cctv',
            'satellite'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'incident_category') THEN
        CREATE TYPE incident_category AS ENUM (
            'pothole',
            'open_manhole',
            'broken_sign',
            'waterlogging',
            'streetlight'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'incident_status') THEN
        CREATE TYPE incident_status AS ENUM (
            'reported',
            'verified',
            'in_progress',
            'resolved'
        );
    END IF;

    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'road_type') THEN
        CREATE TYPE road_type AS ENUM (
            'arterial',
            'collector',
            'local'
        );
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 3. Utility Functions & Triggers
-- ----------------------------------------------------------------------------
-- Function: Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_timestamp_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function: Synchronize Point geometry from latitude and longitude if location is NULL
CREATE OR REPLACE FUNCTION sync_incident_location()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.latitude IS NOT NULL AND NEW.longitude IS NOT NULL THEN
        -- PostGIS Point takes (longitude, latitude) / (X, Y)
        NEW.location = ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ----------------------------------------------------------------------------
-- 4. Table: incidents
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS incidents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source incident_source NOT NULL,
    category incident_category NOT NULL,
    status incident_status NOT NULL DEFAULT 'reported',
    
    -- Geospatial attributes
    location GEOMETRY(Point, 4326),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    address TEXT,
    
    -- Visual & Computer Vision Data
    original_image_url TEXT,
    annotated_image_url TEXT,
    bounding_boxes JSONB NOT NULL DEFAULT '[]'::jsonb,
    visual_embedding VECTOR(512),
    
    -- Prioritization, Traffic & Risk Metrics
    severity_score DOUBLE PRECISION CHECK (severity_score >= 1.0 AND severity_score <= 10.0),
    road_type road_type NOT NULL DEFAULT 'local',
    traffic_density DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    affected_population_score DOUBLE PRECISION,
    estimated_repair_cost DOUBLE PRECISION,
    verification_count INTEGER NOT NULL DEFAULT 1 CHECK (verification_count >= 1),
    pvi_score DOUBLE PRECISION CHECK (pvi_score >= 0.0 AND pvi_score <= 10.0),
    
    -- Auditing
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 5. Table: citizen_feedback
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS citizen_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    incident_id UUID NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    user_upvotes INTEGER NOT NULL DEFAULT 1 CHECK (user_upvotes >= 0),
    comment TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ----------------------------------------------------------------------------
-- 6. Indexes & Performance Optimization
-- ----------------------------------------------------------------------------
-- GIST Spatial Index for fast geospatial proximity/radius searches
CREATE INDEX IF NOT EXISTS idx_incidents_location_gist 
ON incidents USING GIST (location);

-- HNSW Vector Index for sub-millisecond visual similarity search (cosine distance)
CREATE INDEX IF NOT EXISTS idx_incidents_visual_embedding_hnsw 
ON incidents USING hnsw (visual_embedding vector_cosine_ops);

-- B-Tree indexes for status filtering and priority dispatching
CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents (status);
CREATE INDEX IF NOT EXISTS idx_incidents_category ON incidents (category);
CREATE INDEX IF NOT EXISTS idx_incidents_pvi_score_desc ON incidents (pvi_score DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_incidents_created_at_desc ON incidents (created_at DESC);

-- Foreign Key index on citizen feedback
CREATE INDEX IF NOT EXISTS idx_citizen_feedback_incident_id ON citizen_feedback (incident_id);

-- ----------------------------------------------------------------------------
-- 7. Apply Triggers
-- ----------------------------------------------------------------------------
DROP TRIGGER IF EXISTS trg_incidents_updated_at ON incidents;
CREATE TRIGGER trg_incidents_updated_at
BEFORE UPDATE ON incidents
FOR EACH ROW
EXECUTE FUNCTION update_timestamp_column();

DROP TRIGGER IF EXISTS trg_incidents_sync_location ON incidents;
CREATE TRIGGER trg_incidents_sync_location
BEFORE INSERT OR UPDATE ON incidents
FOR EACH ROW
EXECUTE FUNCTION sync_incident_location();
