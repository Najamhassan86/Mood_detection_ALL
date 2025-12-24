from datetime import datetime
from typing import List, Dict, Any

import asyncio
import base64

from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select

from app.db import create_db_and_tables, get_session
from app.models import PersonEmotion
from app.schemas import (
    EmotionsBatchIn,
    EmotionsBatchResponse,
    DashboardSummaryOut,
    EmotionTotalsOut,
    PersonStateOut,
    HealthCheckResponse,
    FrameUpload,
)


# ============================================================
# FastAPI Application Setup
# ============================================================

app = FastAPI(
    title="Emotion Analytics Backend",
    description="Backend API for Jetson-based emotion detection system",
    version="0.1.0",
)

# Configure CORS to allow Next.js dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://emotion-frontend-production.up.railway.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# In-memory video frame store (for MJPEG streaming)
# ============================================================

LATEST_FRAMES: Dict[str, Dict[str, Any]] = {}
# Example:
# LATEST_FRAMES["jetson_1"] = {
#     "frame": b"...jpeg bytes...",
#     "timestamp": datetime(...)
# }


# ============================================================
# Application Lifecycle Events
# ============================================================

@app.on_event("startup")
def on_startup():
    """Initialize database tables on application startup."""
    create_db_and_tables()
    print("✅ Database initialized")


# ============================================================
# Health Check
# ============================================================

@app.get("/", response_model=HealthCheckResponse)
def health_check():
    """
    Health check endpoint.
    Returns status to confirm the API is running.
    """
    return {"status": "ok"}


# ============================================================
# Video Stream Upload + MJPEG Endpoint
# ============================================================

@app.post("/api/stream/frame")
async def upload_frame(payload: FrameUpload):
    """
    Jetson posts base64-encoded JPEG frames here.
    We keep only the latest frame per device in memory.
    """
    try:
        frame_bytes = base64.b64decode(payload.frame_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 frame")

    LATEST_FRAMES[payload.device_id] = {
        "frame": frame_bytes,
        "timestamp": payload.timestamp or datetime.utcnow(),
        "fps": payload.fps,  # ✅ Store FPS from model
    }
    return {"status": "ok"}


async def mjpeg_generator(device_id: str):
    """
    Yield multipart JPEG frames for the given device_id.
    Browser will render this as a live stream (<img src=...>).
    """
    boundary = b"--frame"

    while True:
        data = LATEST_FRAMES.get(device_id)
        if data is not None:
            frame: bytes = data["frame"]
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        # small sleep to avoid 100% CPU
        await asyncio.sleep(0.1)


@app.get("/stream/{device_id}")
async def stream_device(device_id: str):
    """
    MJPEG stream endpoint for frontend.

    Frontend can do:
      <img src="https://emotion-backend-production.up.railway.app/stream/jetson_1" />
    """
    return StreamingResponse(
        mjpeg_generator(device_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ============================================================
# Emotion Batch Ingest (Jetson → DB)
# ============================================================

@app.post("/api/emotions/batch", response_model=EmotionsBatchResponse)
def ingest_emotions_batch(
    batch: EmotionsBatchIn,
    session: Session = Depends(get_session),
):
    """
    Receive batch emotion updates from Jetson device.

    Upsert logic:
    - If (device_id, person_id) exists → update cumulative times
    - If not → insert new record
    """
    # Parse the timestamp from Jetson
    try:
        timestamp = datetime.fromisoformat(batch.timestamp.replace("Z", "+00:00"))
    except Exception:
        # Fallback to current time if parsing fails
        timestamp = datetime.utcnow()

    updated_count = 0

    for person_data in batch.people:
        person_id = person_data.person_id
        cumulative = person_data.cumulative

        # Extract emotion times from cumulative dict
        time_happy = cumulative.get("happy", 0.0)
        time_sad = cumulative.get("sad", 0.0)
        time_angry = cumulative.get("angry", 0.0)
        time_neutral = cumulative.get("neutral", 0.0)

        # Check if record exists
        statement = select(PersonEmotion).where(
            PersonEmotion.device_id == batch.device_id,
            PersonEmotion.person_id == person_id,
        )
        existing = session.exec(statement).first()

        if existing:
            # Update existing record
            existing.time_happy = time_happy
            existing.time_sad = time_sad
            existing.time_angry = time_angry
            existing.time_neutral = time_neutral
            existing.last_seen = timestamp
            session.add(existing)
        else:
            # Insert new record
            new_record = PersonEmotion(
                device_id=batch.device_id,
                person_id=person_id,
                time_happy=time_happy,
                time_sad=time_sad,
                time_angry=time_angry,
                time_neutral=time_neutral,
                last_seen=timestamp,
            )
            session.add(new_record)

        updated_count += 1

    # Commit all changes
    session.commit()

    return EmotionsBatchResponse(status="ok", updated_count=updated_count)


# ============================================================
# Dashboard Summary (DB → Frontend)
# ============================================================

@app.get("/api/dashboard/summary", response_model=DashboardSummaryOut)
def get_dashboard_summary(
    device_id: str = Query(
        default="jetson_1",
        description="Device ID to query",
    ),
    session: Session = Depends(get_session),
):
    """
    Get aggregated dashboard summary for a device.

    Aggregates:
    - Total emotion times across all people
    - Individual person states with current dominant emotion
    - Device metadata and last update time
    """
    # Query all person records for this device
    statement = select(PersonEmotion).where(PersonEmotion.device_id == device_id)
    person_records: List[PersonEmotion] = session.exec(statement).all()

    # Initialize aggregated totals
    total_happy = 0.0
    total_sad = 0.0
    total_angry = 0.0
    total_neutral = 0.0

    # Build current_people list
    current_people: List[PersonStateOut] = []

    for record in person_records:
        # Aggregate totals
        total_happy += record.time_happy
        total_sad += record.time_sad
        total_angry += record.time_angry
        total_neutral += record.time_neutral

        # Determine current/dominant emotion for this person
        emotion_times = {
            "happy": record.time_happy,
            "sad": record.time_sad,
            "angry": record.time_angry,
            "neutral": record.time_neutral,
        }
        current_emotion = max(emotion_times, key=emotion_times.get)

        # If all times are 0, default to neutral
        if (
            record.time_happy == 0
            and record.time_sad == 0
            and record.time_angry == 0
            and record.time_neutral == 0
        ):
            current_emotion = "neutral"

        # Build PersonStateOut
        person_state = PersonStateOut(
            person_id=record.person_id,
            current_emotion=current_emotion,
            time_happy=record.time_happy,
            time_sad=record.time_sad,
            time_angry=record.time_angry,
            time_neutral=record.time_neutral,
            last_seen=record.last_seen.isoformat() + "Z",
        )
        current_people.append(person_state)

    # Build emotion totals
    emotion_totals = EmotionTotalsOut(
        happy=total_happy,
        sad=total_sad,
        angry=total_angry,
        neutral=total_neutral,
    )

    # ✅ Get model FPS from latest frame
    model_fps = 0.0
    latest_frame_data = LATEST_FRAMES.get(device_id)
    if latest_frame_data:
        model_fps = latest_frame_data.get("fps", 0.0)

    # Build final summary
    summary = DashboardSummaryOut(
        device_id=device_id,
        device_name="Entrance Camera",  # TODO: make configurable later
        updated_at=datetime.utcnow().isoformat() + "Z",
        emotion_totals=emotion_totals,
        current_people=current_people,
        model_fps=model_fps,  # ✅ Include model FPS
        frontend_fps=0.0,  # Will be calculated by frontend
    )

    return summary
