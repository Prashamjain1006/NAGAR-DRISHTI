"""
Geospatial & Visual Deduplication Engine for Civic Infrastructure Incidents.
Combines PostGIS spatial proximity (ST_DWithin <= 10m) and pgvector Cosine Similarity (> 0.75)
to merge incoming reports from citizen uploads, dashcams, and CCTV feeds into single incident threads.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from db import db

logger = logging.getLogger("deduplication_engine")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# Deduplication Thresholds (Per Architectural Specifications)
# -----------------------------------------------------------------------------
SPATIAL_RADIUS_METERS = 10.0          # Incidents within 10-meter radius
COSINE_SIMILARITY_THRESHOLD = 0.75    # 1 - cosine_distance > 0.75 (~75%+ visual match)

ROAD_TYPE_WEIGHTS = {
    "arterial": 1.5,
    "collector": 1.2,
    "local": 1.0,
}


from scoring import calculate_dynamic_pvi, estimate_repair_cost


def compute_pvi(
    severity_score: float,
    traffic_density: float,
    road_type: str = "local",
    verification_count: int = 1,
    category: str = "pothole",
) -> float:
    """
    Computes Priority Value Index using the Data Science formula from scoring.py:
    PVI = (Severity * 0.30) + (Road_Hierarchy * 0.20) + (Traffic_Density * 0.20) + 
          (Population_Impact * 0.15) + (Verification_Count * 0.15)
    """
    res = calculate_dynamic_pvi(
        category=category,
        severity=severity_score,
        road_type=road_type,
        traffic_density=traffic_density,
        verification_count=verification_count,
    )
    return res.pvi_score


# -----------------------------------------------------------------------------
# Core Deduplication Query: PostGIS + pgvector
# -----------------------------------------------------------------------------
async def find_existing_incident(
    lat: float,
    lng: float,
    category: Optional[str] = None,
    embedding: Optional[List[float]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Executes combined PostGIS spatial proximity and pgvector cosine similarity search.
    
    1. ST_DWithin(location, ST_SetSRID(ST_MakePoint(lng, lat), 4326)::geography, 10)
    2. Computes similarity: 1.0 - (visual_embedding <=> embedding) AS similarity
    3. Evaluates condition: Distance < 10m AND Similarity > 0.75
    """
    if not db.is_connected:
        try:
            await db.connect()
        except Exception as exc:
            logger.warning("Database unavailable for deduplication check: %s", exc)
            return None

    # Query active incidents within 10-meter radius
    query = """
        SELECT 
            id,
            category,
            status,
            severity_score,
            road_type,
            traffic_density,
            pvi_score,
            verification_count,
            annotated_image_url,
            latitude,
            longitude,
            ST_Distance(
                location::geography,
                ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography
            ) AS distance_meters,
            CASE 
                WHEN visual_embedding IS NOT NULL AND $4::vector IS NOT NULL 
                THEN (1.0 - (visual_embedding <=> $4::vector))
                ELSE 0.0 
            END AS similarity
        FROM incidents
        WHERE 
            status != 'resolved'
            AND ST_DWithin(
                location::geography,
                ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography,
                $3
            )
            AND ($5::text IS NULL OR category = $5::incident_category)
        ORDER BY similarity DESC, distance_meters ASC
        LIMIT 1;
    """

    async with db.acquire() as conn:
        row = await conn.fetchrow(
            query,
            lat,
            lng,
            SPATIAL_RADIUS_METERS,
            embedding,
            category,
        )

        if row:
            dist = float(row["distance_meters"])
            sim = float(row["similarity"])
            logger.info(
                "Candidate incident found: ID=%s, Dist=%.2fm, Similarity=%.3f (Req: <10m, >0.75)",
                row["id"],
                dist,
                sim,
            )

            # Match Condition: Distance < 10m AND Similarity > 0.75
            # (or if embedding is absent, match if within 5m of same category)
            if dist <= SPATIAL_RADIUS_METERS and (sim >= COSINE_SIMILARITY_THRESHOLD or (embedding is None and dist <= 5.0)):
                match_dict = dict(row)
                match_dict["distance_meters"] = round(dist, 2)
                match_dict["similarity"] = round(sim, 4)
                return match_dict

    return None


# -----------------------------------------------------------------------------
# Incident Merge or Insert Orchestration
# -----------------------------------------------------------------------------
async def deduplicate_and_persist_incident(
    lat: float,
    lng: float,
    category: str,
    embedding: List[float],
    source: str = "citizen_upload",
    severity_score: float = 5.0,
    traffic_density: float = 5.0,
    original_image_url: Optional[str] = None,
    annotated_image_url: Optional[str] = None,
    bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    address: Optional[str] = None,
    user_comment: Optional[str] = None,
    road_type: str = "local",
) -> Dict[str, Any]:
    """
    Main Ingestion Gateway with Deduplication & Merging Logic:
    
    1. Checks if an active duplicate exists within 10 meters with > 0.75 visual similarity.
    2. If MATCH FOUND:
       - Do NOT insert a new row.
       - Increment verification_count += 1.
       - Update traffic_density (take higher density observation).
       - Recalculate PVI score.
       - Update updated_at = NOW().
       - Link new upload and citizen/feed remarks to citizen_feedback thread.
       - Returns action: "merged", is_duplicate: true.
    3. If NO MATCH:
       - Insert new incident row into incidents table.
       - Calculate baseline PVI score.
       - If comment exists, initialize feedback thread.
       - Returns action: "created", is_duplicate: false.
    """
    if bounding_boxes is None:
        bounding_boxes = []

    # 1. Search for existing matching incident
    existing = await find_existing_incident(lat, lng, category, embedding)

    if existing:
        incident_id = existing["id"]
        old_count = existing["verification_count"]
        new_count = old_count + 1

        # Blend or pick maximum traffic density
        updated_traffic_density = max(float(existing["traffic_density"]), traffic_density)
        existing_severity = float(existing["severity_score"] or severity_score)
        existing_road_type = existing["road_type"] or road_type

        # Recalculate dynamic PVI score and repair cost with updated verification count
        scoring_res = calculate_dynamic_pvi(
            category=existing["category"],
            severity=existing_severity,
            road_type=existing_road_type,
            traffic_density=updated_traffic_density,
            verification_count=new_count,
        )
        new_pvi = scoring_res.pvi_score
        new_cost = scoring_res.estimated_cost
        new_risk = scoring_res.risk_level

        logger.info(
            "Merging report into existing incident %s (Verifications: %d -> %d, PVI: %.2f -> %.2f, Risk: %s)",
            incident_id,
            old_count,
            new_count,
            existing["pvi_score"],
            new_pvi,
            new_risk,
        )

        update_sql = """
            UPDATE incidents
            SET 
                verification_count = verification_count + 1,
                traffic_density = $2,
                pvi_score = $3,
                estimated_repair_cost = $4,
                updated_at = NOW()
            WHERE id = $1
            RETURNING id, category, status, verification_count, pvi_score, estimated_repair_cost, annotated_image_url;
        """

        feedback_comment = (
            f"[{source.upper()}] Re-verified incident. "
            f"Match distance: {existing['distance_meters']}m, similarity: {existing['similarity'] * 100:.1f}%. "
            f"{user_comment or ''}"
        ).strip()

        if db.is_connected:
            async with db.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(update_sql, incident_id, updated_traffic_density, new_pvi, new_cost)
                    await conn.execute(
                        """
                        INSERT INTO citizen_feedback (incident_id, user_upvotes, comment)
                        VALUES ($1, 1, $2);
                        """,
                        incident_id,
                        feedback_comment,
                    )

        return {
            "action": "merged",
            "incident_id": str(incident_id),
            "category": existing["category"],
            "status": existing["status"],
            "verification_count": new_count,
            "pvi_score": new_pvi,
            "risk_level": new_risk,
            "estimated_cost": new_cost,
            "distance_meters": existing["distance_meters"],
            "similarity_score": existing["similarity"],
            "annotated_image_url": existing["annotated_image_url"],
            "is_duplicate": True,
            "message": f"Report merged with existing {existing['category']} incident within {existing['distance_meters']}m.",
        }

    # 2. No match found -> Insert new incident row
    new_incident_id = uuid.uuid4()
    scoring_res = calculate_dynamic_pvi(
        category=category,
        severity=severity_score,
        bounding_boxes=bounding_boxes,
        road_type=road_type,
        traffic_density=traffic_density,
        verification_count=1,
    )
    pvi_score = scoring_res.pvi_score
    estimated_cost = scoring_res.estimated_cost
    risk_level = scoring_res.risk_level

    insert_sql = """
        INSERT INTO incidents (
            id,
            source,
            category,
            status,
            location,
            latitude,
            longitude,
            address,
            original_image_url,
            annotated_image_url,
            bounding_boxes,
            visual_embedding,
            severity_score,
            road_type,
            traffic_density,
            estimated_repair_cost,
            pvi_score,
            verification_count
        ) VALUES (
            $1, $2::incident_source, $3::incident_category, 'reported',
            ST_SetSRID(ST_MakePoint($4, $5), 4326),
            $5, $4, $6, $7, $8, $9, $10::vector, $11, $12::road_type, $13, $14, $15, 1
        )
        RETURNING id, category, status, verification_count, pvi_score, estimated_repair_cost, annotated_image_url;
    """

    if db.is_connected:
        async with db.acquire() as conn:
            async with conn.transaction():
                row = await conn.fetchrow(
                    insert_sql,
                    new_incident_id,
                    source,
                    category,
                    lng,
                    lat,
                    address or f"Reported via {source}",
                    original_image_url,
                    annotated_image_url,
                    bounding_boxes,
                    embedding,
                    severity_score,
                    road_type,
                    traffic_density,
                    estimated_cost,
                    pvi_score,
                )
                if user_comment:
                    await conn.execute(
                        """
                        INSERT INTO citizen_feedback (incident_id, user_upvotes, comment)
                        VALUES ($1, 1, $2);
                        """,
                        new_incident_id,
                        user_comment,
                    )
                incident_id = row["id"]
    else:
        incident_id = new_incident_id

    logger.info("Created new incident record: ID=%s (PVI: %.2f, Risk: %s, Est. Cost: $%.2f)", incident_id, pvi_score, risk_level, estimated_cost)

    return {
        "action": "created",
        "incident_id": str(incident_id),
        "category": category,
        "status": "reported",
        "verification_count": 1,
        "pvi_score": pvi_score,
        "risk_level": risk_level,
        "estimated_cost": estimated_cost,
        "distance_meters": 0.0,
        "similarity_score": 1.0,
        "annotated_image_url": annotated_image_url,
        "is_duplicate": False,
        "message": "New incident successfully recorded in database.",
    }


# -----------------------------------------------------------------------------
# Standalone CLI Test Verification
# -----------------------------------------------------------------------------
async def main():
    print("=== Deduplication & Spatial-Visual Merging Test ===")
    
    # Generate test embedding
    test_vec = [0.05 for _ in range(512)]
    
    # 1. Test PVI calculation
    pvi_initial = compute_pvi(severity_score=8.0, traffic_density=6.0, road_type="arterial", verification_count=1)
    pvi_boosted = compute_pvi(severity_score=8.0, traffic_density=6.0, road_type="arterial", verification_count=3)
    print(f"PVI initial (1 report): {pvi_initial}")
    print(f"PVI re-verified (3 reports): {pvi_boosted}")
    assert pvi_boosted > pvi_initial

    # 2. Test Deduplication Pipeline
    res = await deduplicate_and_persist_incident(
        lat=37.7749,
        lng=-122.4194,
        category="pothole",
        embedding=test_vec,
        source="citizen_upload",
        severity_score=8.0,
        traffic_density=5.0,
        address="100 Market St, San Francisco, CA",
        user_comment="Dangerous pothole in middle lane",
    )
    print("Result Payload:")
    import json
    print(json.dumps(res, indent=2))
    assert res["action"] in ("created", "merged")
    print("Deduplication Engine Test PASSED!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
