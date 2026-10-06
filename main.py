"""
Civic Infrastructure Incident Platform - Main FastAPI Application.
Integrates:
- CORS middleware for Next.js (http://localhost:3000)
- Static media serving for /uploads
- Real-time WebSockets dispatch (/ws/incidents)
- Prioritized incident discovery and filtering
- City-wide analytics and deduplication cost savings
- Automated startup seeding of 20 realistic incidents with generated annotations
"""

from __future__ import annotations

import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    Body,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont

# Local system modules
from db import db
from deduplication import deduplicate_and_persist_incident
from scoring import calculate_dynamic_pvi, estimate_repair_cost
from vision_engine import (
    ANNOTATED_DIR,
    ORIGINAL_DIR,
    UPLOAD_DIR,
    detect_and_annotate_image,
    draw_bounding_boxes,
    extract_visual_embedding,
)

logger = logging.getLogger("civic_platform")
logging.basicConfig(level=logging.INFO)

# In-memory fallback cache for demo/offline operational readiness
SEED_INCIDENTS_CACHE: List[Dict[str, Any]] = []


# -----------------------------------------------------------------------------
# 1. WebSocket Connection Manager
# -----------------------------------------------------------------------------
class WebSocketManager:
    """Manages active WebSocket subscriber connections and event broadcasting."""

    def __init__(self) -> None:
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("WebSocket subscriber connected. Total active: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket subscriber disconnected. Total active: %d", len(self.active_connections))

    async def broadcast(self, event_type: str, data: Dict[str, Any]) -> None:
        """Broadcasts real-time events (INCIDENT_CREATED, INCIDENT_MERGED, STATUS_UPDATED)."""
        payload = {
            "event": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Broadcasting WS event: %s to %d subscribers", event_type, len(self.active_connections))
        stale_clients = []
        for connection in self.active_connections:
            try:
                await connection.send_json(payload)
            except Exception as exc:
                logger.warning("Error pushing event to WebSocket client: %s", exc)
                stale_clients.append(connection)

        for stale in stale_clients:
            self.disconnect(stale)


ws_manager = WebSocketManager()


# -----------------------------------------------------------------------------
# 2. Automated Seed Data Generator (20 Realistic Incidents)
# -----------------------------------------------------------------------------
SEED_DEFINITIONS = [
    {"cat": "pothole", "lat": 12.9569, "lon": 77.7011, "addr": "Outer Ring Road, Marathahalli Bridge, Bengaluru", "road": "arterial", "sev": 8.8, "traffic": 9.2, "verif": 27, "src": "citizen_upload"},
    {"cat": "open_manhole", "lat": 28.5727, "lon": 77.2588, "addr": "Ring Road, Ashram Chowk Underpass, New Delhi", "road": "arterial", "sev": 9.6, "traffic": 9.4, "verif": 31, "src": "cctv"},
    {"cat": "waterlogging", "lat": 12.9172, "lon": 77.6228, "addr": "Silk Board Junction, Hosur Road, Bengaluru", "road": "arterial", "sev": 9.4, "traffic": 9.8, "verif": 42, "src": "satellite"},
    {"cat": "broken_sign", "lat": 28.6315, "lon": 77.2167, "addr": "Connaught Place Inner Circle, Radial Road 2, New Delhi", "road": "local", "sev": 6.2, "traffic": 7.2, "verif": 9, "src": "dashcam"},
    {"cat": "streetlight", "lat": 13.0418, "lon": 80.2507, "addr": "Anna Salai, Gemini Flyover, Chennai", "road": "arterial", "sev": 5.7, "traffic": 7.8, "verif": 8, "src": "citizen_upload"},
    {"cat": "pothole", "lat": 19.1136, "lon": 72.8697, "addr": "Western Express Highway, Andheri East Flyover, Mumbai", "road": "arterial", "sev": 9.1, "traffic": 9.9, "verif": 35, "src": "dashcam"},
    {"cat": "open_manhole", "lat": 18.5913, "lon": 73.7389, "addr": "Hinjewadi Phase 1, Shivaji Chowk, Pune", "road": "collector", "sev": 8.9, "traffic": 8.6, "verif": 18, "src": "citizen_upload"},
    {"cat": "waterlogging", "lat": 19.0596, "lon": 72.8295, "addr": "SV Road, Bandra West Station, Mumbai", "road": "collector", "sev": 8.6, "traffic": 8.9, "verif": 19, "src": "cctv"},
    {"cat": "pothole", "lat": 17.4504, "lon": 78.3808, "addr": "HITEC City Main Road, Cyber Towers Junction, Hyderabad", "road": "arterial", "sev": 8.4, "traffic": 8.8, "verif": 22, "src": "citizen_upload"},
    {"cat": "streetlight", "lat": 18.9256, "lon": 72.8242, "addr": "Marine Drive Promenade, Nariman Point, Mumbai", "road": "local", "sev": 4.5, "traffic": 6.0, "verif": 5, "src": "citizen_upload"},
    {"cat": "broken_sign", "lat": 22.5448, "lon": 88.3426, "addr": "Victoria Memorial Outer Ring, Kolkata", "road": "local", "sev": 4.8, "traffic": 6.0, "verif": 4, "src": "dashcam"},
    {"cat": "pothole", "lat": 12.9784, "lon": 77.6408, "addr": "100 Feet Road, Indiranagar, Bengaluru", "road": "collector", "sev": 7.4, "traffic": 7.8, "verif": 14, "src": "citizen_upload"},
    {"cat": "waterlogging", "lat": 22.5355, "lon": 88.3995, "addr": "Eastern Metropolitan Bypass, Ruby More, Kolkata", "road": "arterial", "sev": 8.7, "traffic": 8.9, "verif": 25, "src": "satellite"},
    {"cat": "open_manhole", "lat": 19.0178, "lon": 72.8478, "addr": "Dadar TT Circle, Dr Ambedkar Road, Mumbai", "road": "arterial", "sev": 9.2, "traffic": 9.1, "verif": 20, "src": "cctv"},
    {"cat": "streetlight", "lat": 17.4401, "lon": 78.3489, "addr": "Outer Ring Road, Gachibowli Flyover, Hyderabad", "road": "arterial", "sev": 5.4, "traffic": 8.0, "verif": 6, "src": "dashcam"},
    {"cat": "pothole", "lat": 28.4720, "lon": 77.0725, "addr": "MG Road, IFFCO Chowk, Gurugram, Delhi NCR", "road": "arterial", "sev": 8.5, "traffic": 9.0, "verif": 16, "src": "citizen_upload"},
    {"cat": "broken_sign", "lat": 18.5183, "lon": 73.8415, "addr": "FC Road, Deccan Gymkhana, Pune", "road": "local", "sev": 5.0, "traffic": 6.5, "verif": 7, "src": "citizen_upload"},
    {"cat": "pothole", "lat": 13.0037, "lon": 77.6834, "addr": "Old Madras Road, KR Puram Hanging Bridge, Bengaluru", "road": "arterial", "sev": 8.3, "traffic": 9.3, "verif": 21, "src": "dashcam"},
    {"cat": "waterlogging", "lat": 19.0390, "lon": 72.8619, "addr": "Sion Circle, Lal Bahadur Shastri Marg, Mumbai", "road": "arterial", "sev": 8.9, "traffic": 9.5, "verif": 38, "src": "cctv"},
    {"cat": "open_manhole", "lat": 28.5204, "lon": 77.2023, "addr": "Mehrauli-Badarpur Road, Saket Metro, New Delhi", "road": "arterial", "sev": 9.5, "traffic": 9.1, "verif": 29, "src": "citizen_upload"},
]


def generate_seed_annotation_image(category: str, severity: float, filename: str) -> str:
    """Generates an authentic synthetic road image with Pillow-rendered bounding boxes and HUD labels."""
    img = Image.new("RGB", (640, 480), color=(50, 54, 62))
    draw = ImageDraw.Draw(img, "RGBA")

    # Draw simulated road surface asphalt texture
    draw.rectangle([0, 240, 640, 480], fill=(70, 75, 84))
    # Lane divider line
    draw.line([(320, 260), (320, 480)], fill=(220, 180, 50, 200), width=4)

    # Box coordinates
    x0, y0, x1, y1 = 180, 280, 460, 420
    is_high_severity = severity >= 7.5 or category in ("open_manhole", "pothole")
    box_color = (230, 57, 70, 255) if is_high_severity else (244, 162, 97, 255)
    fill_color = (230, 57, 70, 45) if is_high_severity else (244, 162, 97, 45)

    draw.rectangle([x0, y0, x1, y1], outline=box_color, fill=fill_color, width=4)

    # Header label pill
    label_text = f"{category.upper()} {int(min(99, severity * 10 + 2))}%"
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    tb = draw.textbbox((x0, y0), label_text, font=font)
    tw, th = tb[2] - tb[0] + 12, tb[3] - tb[1] + 8
    draw.rectangle([x0, y0 - th, x0 + tw, y0], fill=box_color)
    draw.text((x0 + 6, y0 - th + 4), label_text, fill=(255, 255, 255), font=font)

    dest_path = ANNOTATED_DIR / filename
    img.save(dest_path, format="JPEG", quality=88)
    return f"/uploads/annotated/{filename}"


async def seed_data():
    """Populates 20 realistic pre-annotated incidents across the city on startup."""
    global SEED_INCIDENTS_CACHE
    logger.info("Initializing system seed data...")

    SEED_INCIDENTS_CACHE.clear()

    for idx, item in enumerate(SEED_DEFINITIONS):
        incident_id = uuid.uuid4()
        cat = item["cat"]
        sev = item["sev"]
        road = item["road"]
        traffic = item["traffic"]
        verif = item["verif"]

        # Run Data Science PVI and repair cost scoring
        score_res = calculate_dynamic_pvi(
            category=cat,
            severity=sev,
            road_type=road,
            traffic_density=traffic,
            verification_count=verif,
        )

        annotated_filename = f"seed_{cat}_{idx + 1}.jpg"
        annotated_url = generate_seed_annotation_image(cat, sev, annotated_filename)

        box_payload = [{
            "class": cat,
            "confidence": round(min(0.99, (sev / 10.0) + 0.05), 2),
            "x": 320.0,
            "y": 350.0,
            "width": 280.0,
            "height": 140.0,
        }]

        # Normalized 512-dim embedding
        seed_vec = [round(((i + idx) % 20) * 0.01, 4) for i in range(512)]

        incident_record = {
            "id": str(incident_id),
            "source": item["src"],
            "category": cat,
            "status": "reported" if idx % 4 != 0 else "in_progress",
            "latitude": item["lat"],
            "longitude": item["lon"],
            "address": item["addr"],
            "original_image_url": annotated_url,
            "annotated_image_url": annotated_url,
            "bounding_boxes": box_payload,
            "visual_embedding": seed_vec,
            "severity_score": sev,
            "road_type": road,
            "traffic_density": traffic,
            "affected_population_score": 7.0,
            "estimated_repair_cost": score_res.estimated_cost,
            "pvi_score": score_res.pvi_score,
            "risk_level": score_res.risk_level,
            "verification_count": verif,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        SEED_INCIDENTS_CACHE.append(incident_record)

    # Sort cache descending by PVI
    SEED_INCIDENTS_CACHE.sort(key=lambda x: x["pvi_score"], reverse=True)
    logger.info("Generated %d demo incidents in local memory cache.", len(SEED_INCIDENTS_CACHE))

    # Persist into PostgreSQL if database is connected
    if db.is_connected:
        try:
            async with db.acquire() as conn:
                existing_count = await conn.fetchval("SELECT COUNT(*) FROM incidents;")
                if existing_count == 0:
                    logger.info("Database table 'incidents' is empty. Inserting 20 seed rows...")
                    async with conn.transaction():
                        for inc in SEED_INCIDENTS_CACHE:
                            await conn.execute(
                                """
                                INSERT INTO incidents (
                                    id, source, category, status,
                                    location, latitude, longitude, address,
                                    original_image_url, annotated_image_url,
                                    bounding_boxes, visual_embedding,
                                    severity_score, road_type, traffic_density,
                                    estimated_repair_cost, pvi_score, verification_count
                                ) VALUES (
                                    $1, $2::incident_source, $3::incident_category, $4::incident_status,
                                    ST_SetSRID(ST_MakePoint($5, $6), 4326),
                                    $6, $5, $7,
                                    $8, $9,
                                    $10, $11::vector,
                                    $12, $13::road_type, $14,
                                    $15, $16, $17
                                ) ON CONFLICT (id) DO NOTHING;
                                """,
                                uuid.UUID(inc["id"]),
                                inc["source"],
                                inc["category"],
                                inc["status"],
                                inc["longitude"],
                                inc["latitude"],
                                inc["address"],
                                inc["original_image_url"],
                                inc["annotated_image_url"],
                                inc["bounding_boxes"],
                                inc["visual_embedding"],
                                inc["severity_score"],
                                inc["road_type"],
                                inc["traffic_density"],
                                inc["estimated_repair_cost"],
                                inc["pvi_score"],
                                inc["verification_count"],
                            )
                    logger.info("Successfully seeded 20 incidents into PostgreSQL!")
                else:
                    logger.info("Database already contains %d incidents; skipping database seeding.", existing_count)
        except Exception as exc:
            logger.warning("Could not persist seeds into database: %s. Relying on memory cache.", exc)


# -----------------------------------------------------------------------------
# 3. Lifespan Handler
# -----------------------------------------------------------------------------
@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Orchestrates database connection pool, schema initialization, and demo seeding."""
    logger.info("Starting up Civic Infrastructure Incident Platform...")
    try:
        await db.connect()
        await db.init_schema()
    except Exception as exc:
        logger.warning("Database unavailable on startup (%s). Running with resilient in-memory store.", exc)

    # Seed demo incidents
    await seed_data()

    try:
        yield
    finally:
        logger.info("Shutting down platform...")
        await db.disconnect()


# -----------------------------------------------------------------------------
# 4. FastAPI Application Setup
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Civic Infrastructure Incident & Vision Platform API",
    description="Automated defect detection, PostGIS/pgvector deduplication, and dynamic PVI prioritization.",
    version="1.0.0",
    lifespan=app_lifespan,
)

# CORS configuration for Next.js and client frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static media mount for original and annotated uploads
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


# -----------------------------------------------------------------------------
# 5. WebSockets Real-Time Incident Dispatch
# -----------------------------------------------------------------------------
@app.websocket("/ws/incidents")
async def websocket_incidents_endpoint(websocket: WebSocket):
    """
    Subscribes client to live incident dispatch stream.
    Receives real-time broadcasts for:
    - INCIDENT_CREATED: newly detected defects
    - INCIDENT_MERGED: multi-source duplicate verifications
    - STATUS_UPDATED: repair workflow progression (e.g. reported -> in_progress -> resolved)
    """
    await ws_manager.connect(websocket)
    try:
        # Send initial confirmation handshake
        await websocket.send_json({
            "event": "CONNECTED",
            "message": "Connected to Civic Incident Live Stream",
            "active_subscribers": len(ws_manager.active_connections),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        while True:
            # Keep socket open and handle any incoming client heartbeat pings
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.warning("WebSocket exception: %s", exc)
        ws_manager.disconnect(websocket)


# -----------------------------------------------------------------------------
# 6. REST Endpoints
# -----------------------------------------------------------------------------
@app.get(
    "/api/v1/incidents/prioritized",
    summary="Get prioritized incidents ordered by PVI score DESC",
    tags=["Incidents"],
)
async def get_prioritized_incidents(
    category: Optional[str] = Query(None, description="Filter by defect category"),
    min_pvi: Optional[float] = Query(None, ge=0.0, le=10.0, description="Minimum Priority Value Index"),
    status: Optional[str] = Query(None, description="Filter by status (reported, verified, in_progress, resolved)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> Dict[str, Any]:
    """
    Returns dynamically prioritized infrastructure issues sorted by PVI score descending.
    Applies optional filters for category, minimum PVI threshold, and repair status.
    """
    if db.is_connected:
        try:
            async with db.acquire() as conn:
                clauses = ["1=1"]
                params: List[Any] = []

                if category:
                    params.append(category)
                    clauses.append(f"category = ${len(params)}::incident_category")
                if min_pvi is not None:
                    params.append(min_pvi)
                    clauses.append(f"pvi_score >= ${len(params)}")
                if status:
                    params.append(status)
                    clauses.append(f"status = ${len(params)}::incident_status")

                where_stmt = " AND ".join(clauses)

                query = f"""
                    SELECT 
                        id, source, category, status,
                        latitude, longitude, address,
                        original_image_url, annotated_image_url,
                        bounding_boxes, severity_score, road_type,
                        traffic_density, estimated_repair_cost, pvi_score,
                        verification_count, created_at, updated_at
                    FROM incidents
                    WHERE {where_stmt}
                    ORDER BY pvi_score DESC NULLS LAST
                    LIMIT ${len(params) + 1} OFFSET ${len(params) + 2};
                """
                params.extend([limit, offset])
                rows = await conn.fetch(query, *params)

                results = []
                for r in rows:
                    rec = dict(r)
                    rec["id"] = str(rec["id"])
                    rec["pvi_score"] = float(rec["pvi_score"] or 0.0)
                    rec["estimated_repair_cost"] = float(rec["estimated_repair_cost"] or 0.0)
                    # Assign risk level dynamically
                    pvi = rec["pvi_score"]
                    rec["risk_level"] = "CRITICAL" if pvi >= 8.0 else "HIGH" if pvi >= 6.5 else "MEDIUM" if pvi >= 4.5 else "LOW"
                    results.append(rec)

                total_count = await conn.fetchval(f"SELECT COUNT(*) FROM incidents WHERE {where_stmt};", *params[:-2])

                return {
                    "total": total_count,
                    "limit": limit,
                    "offset": offset,
                    "incidents": results,
                }
        except Exception as exc:
            logger.warning("Database query error: %s. Falling back to in-memory store.", exc)

    # In-Memory Cache Fallback
    filtered = list(SEED_INCIDENTS_CACHE)
    if category:
        filtered = [x for x in filtered if x["category"] == category]
    if min_pvi is not None:
        filtered = [x for x in filtered if x["pvi_score"] >= min_pvi]
    if status:
        filtered = [x for x in filtered if x["status"] == status]

    filtered.sort(key=lambda x: x["pvi_score"], reverse=True)
    paged = filtered[offset : offset + limit]

    return {
        "total": len(filtered),
        "limit": limit,
        "offset": offset,
        "incidents": paged,
    }


@app.get(
    "/api/v1/analytics/summary",
    summary="City-wide defect analytics, repair cost estimations & deduplication savings",
    tags=["Analytics"],
)
async def get_analytics_summary() -> Dict[str, Any]:
    """
    Computes real-time executive dashboard metrics:
    - Total defects & cost estimation
    - Citizen vs. automated ingestion breakdown
    - Deduplication savings (duplicate tickets merged & budget saved)
    - Risk severity and category distributions
    """
    dataset: List[Dict[str, Any]] = []

    if db.is_connected:
        try:
            async with db.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT 
                        id, source, category, status, pvi_score, 
                        estimated_repair_cost, verification_count 
                    FROM incidents;
                    """
                )
                dataset = [dict(r) for r in rows]
        except Exception as exc:
            logger.warning("Failed querying analytics from database: %s", exc)

    if not dataset:
        dataset = SEED_INCIDENTS_CACHE

    total_defects = len(dataset)
    total_cost = sum(float(x.get("estimated_repair_cost") or 0.0) for x in dataset)

    # Source breakdown
    citizen_reports = sum(1 for x in dataset if str(x.get("source")) == "citizen_upload")
    automated_reports = total_defects - citizen_reports

    sources_dict: Dict[str, int] = {}
    categories_dict: Dict[str, int] = {}
    risk_dict = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}

    total_verifications = 0
    for x in dataset:
        src = str(x.get("source") or "citizen_upload")
        sources_dict[src] = sources_dict.get(src, 0) + 1

        cat = str(x.get("category") or "pothole")
        categories_dict[cat] = categories_dict.get(cat, 0) + 1

        pvi = float(x.get("pvi_score") or 0.0)
        if pvi >= 8.0:
            risk_dict["CRITICAL"] += 1
        elif pvi >= 6.5:
            risk_dict["HIGH"] += 1
        elif pvi >= 4.5:
            risk_dict["MEDIUM"] += 1
        else:
            risk_dict["LOW"] += 1

        total_verifications += int(x.get("verification_count") or 1)

    # Deduplication Savings Calculations:
    # Each merged duplicate eliminates a redundant field inspection (~$120 / inspection + 2 labor hours)
    duplicate_reports_merged = max(0, total_verifications - total_defects)
    inspection_cost_per_ticket = 120.0  # Industry standard municipal dispatch cost
    labor_hours_per_ticket = 2.0
    dedup_cost_saved = duplicate_reports_merged * inspection_cost_per_ticket
    dedup_hours_saved = duplicate_reports_merged * labor_hours_per_ticket

    return {
        "total_defects": total_defects,
        "total_estimated_cost_usd": round(total_cost, 2),
        "reporting_sources": {
            "citizen_reports": citizen_reports,
            "automated_reports": automated_reports,
            "breakdown": sources_dict,
        },
        "deduplication_savings": {
            "duplicate_reports_merged": duplicate_reports_merged,
            "estimated_cost_saved_usd": round(dedup_cost_saved, 2),
            "labor_hours_saved": round(dedup_hours_saved, 1),
            "efficiency_multiplier": round(total_verifications / max(1, total_defects), 2),
        },
        "risk_distribution": risk_dict,
        "category_distribution": categories_dict,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/api/v1/upload-photo",
    summary="Upload citizen defect photo, detect defects, deduplicate & broadcast event",
    tags=["Incidents"],
)
async def upload_photo(
    file: UploadFile = File(..., description="Defect photo (JPEG/PNG)"),
    latitude: float = Form(..., description="Latitude coordinate"),
    longitude: float = Form(..., description="Longitude coordinate"),
    address: Optional[str] = Form(None, description="Human readable address"),
    user_comment: Optional[str] = Form(None, description="Citizen comment"),
) -> Dict[str, Any]:
    """
    Full automated pipeline:
    1. Reads binary photo.
    2. Runs defect detection (YOLO / Roboflow).
    3. Overlays bounding boxes with severity color coding.
    4. Extracts 512-dim CLIP embedding.
    5. Deduplicates against PostGIS (10m radius) & pgvector (Cosine Sim > 0.75).
    6. Persists to PostgreSQL 16.
    7. Broadcasts WebSocket event to live dashboards.
    """
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save original image
    orig_filename = f"orig_{uuid.uuid4().hex}.jpg"
    orig_path = ORIGINAL_DIR / orig_filename
    orig_path.write_bytes(image_bytes)
    original_image_url = f"/uploads/original/{orig_filename}"

    # 1. Defect Detection
    predictions = await detect_and_annotate_image(image_bytes)
    valid_categories = {"pothole", "open_manhole", "broken_sign", "waterlogging", "streetlight"}
    primary_pred = predictions[0] if predictions else None
    detected_class = primary_pred["class"] if primary_pred else "pothole"
    category = detected_class if detected_class in valid_categories else "pothole"

    # 2. Draw Bounding Boxes
    annotated_image_url = draw_bounding_boxes(image_bytes, predictions)

    # 3. Extract 512-dim Embedding
    visual_embedding = extract_visual_embedding(image_bytes, bounding_box=primary_pred)

    top_conf = float(primary_pred["confidence"]) if primary_pred else 0.8
    severity_score = round(max(1.0, min(10.0, top_conf * 10.0)), 1)
    traffic_density = 5.0

    # 4. Centralized Spatial-Visual Deduplication & Database Persistence
    result = await deduplicate_and_persist_incident(
        lat=latitude,
        lng=longitude,
        category=category,
        embedding=visual_embedding,
        source="citizen_upload",
        severity_score=severity_score,
        traffic_density=traffic_density,
        original_image_url=original_image_url,
        annotated_image_url=annotated_image_url,
        bounding_boxes=predictions,
        address=address or "Citizen Mobile Upload",
        user_comment=user_comment,
        road_type="local",
    )

    # Also sync into in-memory fallback list
    if result["action"] == "created":
        SEED_INCIDENTS_CACHE.append({
            "id": result["incident_id"],
            "source": "citizen_upload",
            "category": result["category"],
            "status": "reported",
            "latitude": latitude,
            "longitude": longitude,
            "address": address or "Citizen Mobile Upload",
            "original_image_url": original_image_url,
            "annotated_image_url": annotated_image_url,
            "bounding_boxes": predictions,
            "severity_score": severity_score,
            "pvi_score": result["pvi_score"],
            "risk_level": result["risk_level"],
            "estimated_repair_cost": result["estimated_cost"],
            "verification_count": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        event_name = "INCIDENT_CREATED"
    else:
        # Update existing in memory
        for item in SEED_INCIDENTS_CACHE:
            if item["id"] == result["incident_id"]:
                item["verification_count"] = result["verification_count"]
                item["pvi_score"] = result["pvi_score"]
                item["risk_level"] = result["risk_level"]
                break
        event_name = "INCIDENT_MERGED"

    # 5. Broadcast real-time WebSocket event
    await ws_manager.broadcast(event_name, result)

    return {
        "incident_id": result["incident_id"],
        "pvi_score": result["pvi_score"],
        "risk_level": result["risk_level"],
        "estimated_cost": result["estimated_cost"],
        "category": result["category"],
        "annotated_image_url": result["annotated_image_url"],
        "status": result["status"],
        "is_duplicate": result["is_duplicate"],
        "action": result["action"],
    }


@app.patch(
    "/api/v1/incidents/{incident_id}/status",
    summary="Update incident workflow status and broadcast change",
    tags=["Incidents"],
)
async def update_incident_status(
    incident_id: str,
    new_status: str = Query(..., description="Target status: reported, verified, in_progress, resolved"),
) -> Dict[str, Any]:
    """Updates the status of an incident and broadcasts STATUS_UPDATED across WebSockets."""
    valid_statuses = {"reported", "verified", "in_progress", "resolved"}
    if new_status not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status: '{new_status}'. Must be one of {valid_statuses}",
        )

    # 1. Update in PostgreSQL if connected
    if db.is_connected:
        try:
            async with db.acquire() as conn:
                async with conn.transaction():
                    await conn.execute(
                        "UPDATE incidents SET status = $1::incident_status, updated_at = NOW() WHERE id = $2;",
                        new_status,
                        uuid.UUID(incident_id),
                    )
        except Exception as exc:
            logger.warning("Failed updating status in database: %s", exc)

    # 2. Update in in-memory cache
    found = False
    for inc in SEED_INCIDENTS_CACHE:
        if inc["id"] == incident_id:
            inc["status"] = new_status
            inc["updated_at"] = datetime.now(timezone.utc).isoformat()
            found = True
            break

    event_payload = {
        "incident_id": incident_id,
        "status": new_status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 3. Broadcast STATUS_UPDATED event to WebSocket subscribers
    await ws_manager.broadcast("STATUS_UPDATED", event_payload)

    return {"success": True, "incident_id": incident_id, "status": new_status}


@app.post(
    "/api/v1/ingest/frame",
    summary="Ingest simulated or live dashcam frame from municipal fleet / city buses",
    tags=["Ingestion"],
)
async def ingest_frame(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """
    Ingests live or simulated fleet dashcam frame:
    - Extracts coordinates (along MUNI bus corridors)
    - Runs defect detection & annotation
    - Executes PostGIS + pgvector deduplication
    - Persists and broadcasts INCIDENT_CREATED or INCIDENT_MERGED to WebSockets
    """
    vehicle_id = payload.get("vehicle_id", f"MUNI-BUS-{uuid.uuid4().hex[:4].upper()}")
    route_name = payload.get("route_name", "BMTC Route 500-D (Outer Ring Road Express)")
    category = payload.get("category", "pothole")
    severity = float(payload.get("severity", 8.2))
    
    # Coordinates along Geary / Van Ness corridor
    jitter_lat = (uuid.uuid4().int % 200 - 100) * 0.0001
    jitter_lon = (uuid.uuid4().int % 200 - 100) * 0.0001
    lat = float(payload.get("latitude", 12.9569 + jitter_lat))
    lon = float(payload.get("longitude", 77.7011 + jitter_lon))
    address = payload.get("address", f"{route_name} near 22nd Ave")

    # Generate synthetic defect annotation image
    annotated_filename = f"dashcam_{category}_{uuid.uuid4().hex[:8]}.jpg"
    annotated_url = generate_seed_annotation_image(category, severity, annotated_filename)

    predictions = [{
        "class": category,
        "confidence": 0.94,
        "x": 320.0,
        "y": 300.0,
        "width": 260.0,
        "height": 130.0,
    }]
    embedding = [round(((i + int(severity * 10)) % 25) * 0.01, 4) for i in range(512)]

    result = await deduplicate_and_persist_incident(
        lat=lat,
        lng=lon,
        category=category,
        embedding=embedding,
        source="dashcam",
        severity_score=severity,
        traffic_density=8.5,
        original_image_url=annotated_url,
        annotated_image_url=annotated_url,
        bounding_boxes=predictions,
        address=address,
        user_comment=f"Automated fleet scan via {vehicle_id} ({route_name})",
        road_type="arterial",
    )

    event_name = "INCIDENT_CREATED" if result["action"] == "created" else "INCIDENT_MERGED"

    if result["action"] == "created":
        SEED_INCIDENTS_CACHE.append({
            "id": result["incident_id"],
            "source": "dashcam",
            "category": category,
            "status": "reported",
            "latitude": lat,
            "longitude": lon,
            "address": address,
            "original_image_url": annotated_url,
            "annotated_image_url": annotated_url,
            "bounding_boxes": predictions,
            "severity_score": severity,
            "pvi_score": result["pvi_score"],
            "risk_level": result["risk_level"],
            "estimated_repair_cost": result["estimated_cost"],
            "verification_count": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
    else:
        for item in SEED_INCIDENTS_CACHE:
            if item["id"] == result["incident_id"]:
                item["verification_count"] = result["verification_count"]
                item["pvi_score"] = result["pvi_score"]
                item["risk_level"] = result["risk_level"]
                break

    await ws_manager.broadcast(event_name, result)
    return result


@app.post(
    "/api/v1/simulate/emergency",
    summary="Simulate high-priority citizen emergency defect report",
    tags=["Simulation"],
)
async def simulate_emergency(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Simulates an urgent open manhole report in a dense transit/downtown zone."""
    category = payload.get("category", "open_manhole")
    severity = float(payload.get("severity", 9.8))
    
    jitter_lat = (uuid.uuid4().int % 100 - 50) * 0.0001
    jitter_lon = (uuid.uuid4().int % 100 - 50) * 0.0001
    lat = float(payload.get("latitude", 28.5727 + jitter_lat))
    lon = float(payload.get("longitude", 77.2588 + jitter_lon))
    address = payload.get("address", "Ring Road, Ashram Chowk Underpass, New Delhi (Urgent)")

    annotated_filename = f"emergency_{category}_{uuid.uuid4().hex[:8]}.jpg"
    annotated_url = generate_seed_annotation_image(category, severity, annotated_filename)

    predictions = [{
        "class": category,
        "confidence": 0.99,
        "x": 320.0,
        "y": 320.0,
        "width": 300.0,
        "height": 180.0,
    }]
    embedding = [round(((i + 77) % 30) * 0.01, 4) for i in range(512)]

    result = await deduplicate_and_persist_incident(
        lat=lat,
        lng=lon,
        category=category,
        embedding=embedding,
        source="citizen_upload",
        severity_score=severity,
        traffic_density=9.5,
        original_image_url=annotated_url,
        annotated_image_url=annotated_url,
        bounding_boxes=predictions,
        address=address,
        user_comment="EMERGENCY CITIZEN REPORT: Missing iron manhole cover in crosswalk!",
        road_type="arterial",
    )

    event_name = "INCIDENT_CREATED" if result["action"] == "created" else "INCIDENT_MERGED"

    if result["action"] == "created":
        SEED_INCIDENTS_CACHE.append({
            "id": result["incident_id"],
            "source": "citizen_upload",
            "category": category,
            "status": "reported",
            "latitude": lat,
            "longitude": lon,
            "address": address,
            "original_image_url": annotated_url,
            "annotated_image_url": annotated_url,
            "bounding_boxes": predictions,
            "severity_score": severity,
            "pvi_score": result["pvi_score"],
            "risk_level": result["risk_level"],
            "estimated_repair_cost": result["estimated_cost"],
            "verification_count": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

    await ws_manager.broadcast(event_name, result)
    return result


@app.get("/health", tags=["Monitoring"])
async def health_check_endpoint():
    """Health check ping verifying database pool, WebSocket listeners, and cache status."""
    db_health = await db.health_check()
    return {
        "status": "online",
        "database": db_health,
        "active_websocket_subscribers": len(ws_manager.active_connections),
        "cached_incidents_count": len(SEED_INCIDENTS_CACHE),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# -----------------------------------------------------------------------------
# Standalone CLI Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
