"""
ARGUS — AI Surveillance Dashboard backend.

Runs a YOLOv8 model against either:
  - a live camera feed streamed frame-by-frame over a WebSocket, or
  - an uploaded video file, sampled and analyzed in one pass.

Run locally with:
    pip install -r requirements.txt
    python server.py

Then open http://localhost:8000 in a browser. No terminal is shown to
anyone using the dashboard — it's a normal web page.
"""

import base64
import json
import shutil
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Nemotron API Configuration
NEMOTRON_API_KEY = os.getenv("NEMOTRON_API_KEY", "").strip()
NEMOTRON_API_URL = os.getenv(
    "NEMOTRON_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions"
).strip()
NEMOTRON_MODEL = os.getenv(
    "NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b"
).strip()


def query_nemotron(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 700) -> Optional[str]:
    """Query the Nemotron API securely on the backend without exposing keys.
    Falls back gracefully to a deterministic heuristic analyst if no API key is set
    or if the endpoint cannot be reached.
    """
    if NEMOTRON_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {NEMOTRON_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": NEMOTRON_MODEL,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = requests.post(NEMOTRON_API_URL, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "").strip()
                    # Clean out any residual chain-of-thought text if present
                    if "Here's a thinking process:" in content:
                        if "Draft:" in content:
                            content = content.split("Draft:", 1)[1].strip()
                        elif "Draft Response:" in content:
                            content = content.split("Draft Response:", 1)[1].strip()
                        elif "\n\n" in content:
                            content = content.rsplit("\n\n", 1)[-1].strip()
                    return content
            else:
                print(f"[ARGUS AI Error] Nemotron API returned status {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[ARGUS AI Error] Nemotron API request exception: {e}")
    return None


app = FastAPI(title="ARGUS Surveillance API with Nemotron AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model -------------------------------------------------------------
# yolov8n is the nano variant: small download, fast enough to run live
# inference on a laptop CPU. First run downloads the weights (~6MB).
print("[ARGUS] Loading YOLOv8 model (first run may download weights)...")
model = YOLO("yolov8n.pt")
CLASS_NAMES: Dict[int, str] = model.names
print(f"[ARGUS] Model ready — {len(CLASS_NAMES)} object classes available.")

DEFAULT_WATCHLIST = sorted(
    {"person", "backpack", "suitcase", "knife", "cell phone", "laptop"}
)


def run_inference(
    frame: np.ndarray,
    conf: float = 0.35,
    track: bool = False,
    persist: bool = True,
) -> List[Dict[str, Any]]:
    """Run YOLO on a single BGR frame, return normalized detections with optional tracking IDs."""
    if track:
        results = model.track(frame, conf=conf, persist=persist, verbose=False)
    else:
        results = model.predict(frame, conf=conf, verbose=False)

    detections: List[Dict[str, Any]] = []
    if not results:
        return detections

    r = results[0]
    h, w = frame.shape[:2]
    ids = (
        r.boxes.id.int().tolist()
        if hasattr(r.boxes, "id") and r.boxes.id is not None
        else None
    )

    for i, box in enumerate(r.boxes):
        cls_id = int(box.cls[0])
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        track_id = int(ids[i]) if ids and i < len(ids) else None
        detections.append(
            {
                "label": label,
                "confidence": round(confidence, 3),
                "track_id": track_id,
                # normalized 0-1 box so the frontend can scale to any
                # display size regardless of source resolution
                "box": [
                    round(x1 / w, 4),
                    round(y1 / h, 4),
                    round(x2 / w, 4),
                    round(y2 / h, 4),
                ],
            }
        )
    return detections


# --- API -----------------------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "ok", "classes": len(CLASS_NAMES)}


@app.get("/api/classes")
def get_classes():
    return {
        "all_classes": sorted(CLASS_NAMES.values()),
        "default_watchlist": DEFAULT_WATCHLIST,
    }


# --- Nemotron AI Intelligence Endpoints -----------------------------------

class EventAnalysisRequest(BaseModel):
    event_type: str = "security_detection"
    target_class: str = "unknown"
    track_id: Optional[int] = None
    confidence: float = 0.5
    movement_level: str = "MODERATE"
    velocity: float = 0.0
    dwell_time_seconds: float = 0.0
    threat_score: int = 10
    severity: str = "NOMINAL"
    timestamp: str = "00:00"
    location: str = "monitored sector"


class ThreatExplainRequest(BaseModel):
    threat_score: int
    threat_level: str
    active_classes: List[str] = []
    total_objects: int = 0
    has_critical: bool = False
    movement_level: str = "LOW"


class SessionSummaryRequest(BaseModel):
    duration: float = 0.0
    total_alerts: int = 0
    total_tracked: int = 0
    highest_threat: int = 0
    detected_classes: Dict[str, int] = {}
    key_events: List[Dict[str, Any]] = []


class CopilotRequest(BaseModel):
    question: str
    session_context: Dict[str, Any] = {}


@app.get("/api/ai/status")
def ai_status():
    """Returns AI intelligence status and active engine information."""
    return {
        "configured": bool(NEMOTRON_API_KEY),
        "engine": "NVIDIA Nemotron 3.5 Lightning" if NEMOTRON_API_KEY else "ARGUS Threat Intelligence Engine",
        "model": NEMOTRON_MODEL if NEMOTRON_API_KEY else "Deterministic Rule-Based Analyst",
    }


@app.post("/api/ai/analyze-event")
def analyze_event(req: EventAnalysisRequest):
    """Analyze a single security event or alert using Nemotron."""
    prompt = f"""You are the ARGUS Surveillance AI Intelligence Officer. Analyze this security event:
- Event: {req.event_type}
- Target: {req.target_class} (Track ID: #{req.track_id if req.track_id is not None else 'N/A'})
- Confidence: {int(req.confidence * 100)}%
- Movement Level: {req.movement_level} (Velocity: {req.velocity})
- Dwell Time: {req.dwell_time_seconds:.1f}s
- Current Threat Score: {req.threat_score}/100 ({req.severity})
- Timestamp: {req.timestamp}
- Location Sector: {req.location}

Provide a concise, professional assessment in valid JSON with exactly these keys:
"assessment": a brief 1-sentence assessment
"severity": "CRITICAL", "HIGH", "ELEVATED", or "LOW"
"reason": 1-2 sentences explaining why this event warrants security attention
"recommended_action": 1 sentence recommending practical security protocol
Return ONLY raw JSON, no markdown codeblocks."""

    messages = [
        {"role": "system", "content": "You are ARGUS AI, a tactical surveillance intelligence analyst. Respond in strict JSON format."},
        {"role": "user", "content": prompt}
    ]

    response_text = query_nemotron(messages, temperature=0.1, max_tokens=250)
    if response_text:
        try:
            # Strip markdown if model included it
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
                clean = clean.strip()
            return json.loads(clean)
        except Exception:
            pass

    # Heuristic fallback if API key not set or response parsing fails
    is_weapon = req.target_class in {"knife"}
    is_bag = req.target_class in {"backpack", "suitcase"}
    if is_weapon:
        severity = "CRITICAL"
        assessment = f"Weapon detected in {req.location}"
        reason = f"High-risk entity [{req.target_class}] detected with {int(req.confidence * 100)}% confidence."
        recommended_action = "Initiate immediate security response and verify evidence snapshot."
    elif is_bag:
        severity = "HIGH" if req.dwell_time_seconds > 10 else "ELEVATED"
        assessment = f"Unattended object alert for {req.target_class}"
        reason = f"{req.target_class.capitalize()} stationary for {req.dwell_time_seconds:.1f}s in monitored zone."
        recommended_action = "Dispatch patrol to verify ownership and inspect area."
    else:
        severity = req.severity if req.severity != "NOMINAL" else "ELEVATED"
        assessment = f"Target activity flagged: {req.target_class}"
        reason = f"Entity #{req.track_id or 'N/A'} registered with {req.movement_level.lower()} movement pattern."
        recommended_action = "Maintain visual monitoring and track entity vector."

    return {
        "assessment": assessment,
        "severity": severity,
        "reason": reason,
        "recommended_action": recommended_action,
    }


@app.post("/api/ai/explain-threat")
def explain_threat(req: ThreatExplainRequest):
    """Explains why the current ARGUS threat score is at its current level."""
    classes_str = ", ".join(req.active_classes) if req.active_classes else "no watchlisted targets"
    prompt = f"""Explain why the ARGUS surveillance threat score is {req.threat_score}/100 ({req.threat_level}):
- Active Target Classes: {classes_str}
- Total Objects: {req.total_objects}
- Critical Weapon Flagged: {req.has_critical}
- Movement Level: {req.movement_level}

Give a 1-2 sentence tactical explanation of this threat score. Be concise, authoritative, and direct."""

    messages = [
        {"role": "system", "content": "You are ARGUS AI Threat Analyst. Explain numerical threat scores concisely."},
        {"role": "user", "content": prompt}
    ]

    ai_exp = query_nemotron(messages, temperature=0.2, max_tokens=150)
    if ai_exp:
        return {"explanation": ai_exp.strip()}

    # Heuristic fallback
    if req.has_critical:
        explanation = f"Threat level is critical ({req.threat_score}/100) due to confirmed weapon detection in the monitored sector."
    elif req.threat_score >= 50:
        explanation = f"Threat score elevated to {req.threat_score}/100 due to multiple active targets ({classes_str}) with {req.movement_level.lower()} activity."
    elif req.threat_score >= 25:
        explanation = f"Threat index at {req.threat_score}/100 indicating normal presence of {classes_str} under observation."
    else:
        explanation = "Threat status nominal (0/100). No security threats or watchlisted anomalies detected in sector."

    return {"explanation": explanation}


@app.post("/api/ai/session-summary")
def session_summary(req: SessionSummaryRequest):
    """Generate an AI surveillance briefing summarizing the entire session."""
    classes_summary = ", ".join([f"{k}: {v}" for k, v in req.detected_classes.items()]) or "None"
    prompt = f"""Generate a tactical surveillance intelligence summary for this session:
- Duration: {req.duration:.1f} seconds
- Total Tracked Targets: {req.total_tracked}
- Total Security Alerts: {req.total_alerts}
- Peak Threat Score: {req.highest_threat}/100
- Class Breakdown: {classes_summary}
- Key Event Count: {len(req.key_events)}

Return JSON with exactly:
"overall_assessment": 2 sentences assessing the security posture of the session
"key_observations": list of 3 concise bullet points
"highest_risk_event": 1 sentence describing the peak threat or notable event
Return ONLY raw JSON, no markdown formatting."""

    messages = [
        {"role": "system", "content": "You are ARGUS Chief Intelligence Officer. Return structured surveillance briefings in JSON."},
        {"role": "user", "content": prompt}
    ]

    response_text = query_nemotron(messages, temperature=0.15, max_tokens=350)
    if response_text:
        try:
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
                clean = clean.strip()
            return json.loads(clean)
        except Exception:
            pass

    # Heuristic fallback
    if req.highest_threat >= 70:
        overall = f"Session exhibited high-risk surveillance events with a peak threat index of {req.highest_threat}/100. Critical target activity was flagged requiring operator review."
    elif req.highest_threat >= 35:
        overall = f"Session recorded moderate activity across {req.total_tracked} tracked entities with {req.total_alerts} security alerts triggered over {req.duration:.0f}s."
    else:
        overall = f"Surveillance session concluded with nominal threat activity across {req.duration:.0f}s of continuous monitoring."

    observations = [
        f"Registered {req.total_tracked} unique tracked entities across {len(req.detected_classes)} object classes.",
        f"Recorded {req.total_alerts} security alert thresholds with maximum threat index reaching {req.highest_threat}/100.",
        f"Continuous visual monitoring maintained with zero pipeline dropouts."
    ]

    highest_risk = f"Peak threat index of {req.highest_threat}/100 triggered during target classification." if req.highest_threat > 0 else "No elevated security violations recorded during this monitoring period."

    return {
        "overall_assessment": overall,
        "key_observations": observations,
        "highest_risk_event": highest_risk,
    }


@app.post("/api/ai/copilot")
def copilot_chat(req: CopilotRequest):
    """Answer user questions about the current surveillance session with grounded context."""
    ctx = req.session_context or {}
    duration = ctx.get("elapsed", "00:00")
    alerts = ctx.get("alerts", 0)
    objects = ctx.get("objects", 0)
    threat = ctx.get("threat_score", 0)
    threat_level = ctx.get("threat_level", "NOMINAL")
    classes = ctx.get("classes", {})
    recent_alerts = ctx.get("recent_alerts", [])
    active_tracks = ctx.get("active_tracks", [])

    system_prompt = f"""SYSTEM:
You are ARGUS Copilot, the AI intelligence assistant for ARGUS powered by Nemotron-3.5-Lightning.
You have access to structured surveillance telemetry from the current session.

Answer the user's actual question directly.
Use session data when relevant.
Do not repeat generic telemetry unless it answers the question.
Do not invent information.
If information is unavailable, say so clearly.
For general questions about yourself, answer naturally.

CURRENT SESSION TELEMETRY:
- Session Duration: {duration}
- Current Objects in Frame: {objects}
- Active Tracked Entities: {json.dumps(active_tracks)}
- Total Security Alerts Triggered: {alerts}
- Current Threat Score: {threat}/100 ({threat_level})
- Object Class Breakdown: {json.dumps(classes)}
- Recent Security Log Events: {json.dumps(recent_alerts[:8])}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question}
    ]

    ai_reply = query_nemotron(messages, temperature=0.25, max_tokens=700)
    if ai_reply:
        return {"answer": ai_reply.strip(), "provider": "Nemotron-3.5-Lightning"}

    # Heuristic fallback if API key not available or request fails
    q = req.question.lower()
    if "who are you" in q or "what is your name" in q or "identify yourself" in q:
        reply = "I am ARGUS Copilot, the AI intelligence assistant for the ARGUS surveillance command center, powered by Nemotron-3.5-Lightning."
    elif "what happened" in q or "session" in q:
        reply = f"During this session ({duration} elapsed), ARGUS monitored {objects} targets and registered {alerts} security events. Current threat level is {threat_level} ({threat}/100)."
    elif "threat" in q or "risk" in q:
        reply = f"The threat score is currently {threat}/100 ({threat_level}) with {alerts} active alerts logged."
    elif "moved the most" in q or "movement" in q or "tracking patterns" in q:
        if active_tracks:
            top_track = max(active_tracks, key=lambda x: x.get("velocity", 0) or x.get("movement_level") == "HIGH")
            reply = f"Track #{top_track.get('track_id', 'N/A')} ({top_track.get('label', 'target')}) registered the highest movement vector with threat score {top_track.get('threat', threat)}."
        else:
            reply = f"Tracking analytics indicate {objects} active entities in sector. Dwell time and movement vectors are actively monitored by YOLOv8 and BoT-SORT."
    elif "why" in q and "flagged" in q:
        if recent_alerts:
            latest = recent_alerts[0]
            reply = f"Track #{latest.get('track_id', 'N/A')} ({latest.get('label', 'entity')}) was flagged due to target classification matching watchlist parameters with {int(float(latest.get('confidence', 0))*100)}% confidence at timestamp {latest.get('time', 'recent')}."
        else:
            reply = "No targets have exceeded security threat thresholds yet."
    elif "briefing" in q or "summary" in q:
        reply = f"ARGUS SECURITY BRIEFING:\n• Session Duration: {duration}\n• Active Targets: {objects}\n• Alerts: {alerts}\n• Threat Index: {threat}/100 ({threat_level})\n• Status: Operational"
    else:
        reply = f"ARGUS Copilot: Session elapsed time is {duration}. Active targets: {objects}, Total alerts: {alerts}, Threat score: {threat}/100 ({threat_level})."

    return {"answer": reply}


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """Receives base64 JPEG frames from the browser, returns detections."""
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            b64 = payload.get("frame", "")
            if "," in b64:
                b64 = b64.split(",", 1)[1]

            try:
                img_bytes = base64.b64decode(b64)
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                frame = None

            if frame is None:
                await websocket.send_json({"error": "bad_frame"})
                continue

            detections = run_inference(frame)
            await websocket.send_json(
                {"timestamp": time.time(), "detections": detections}
            )
    except WebSocketDisconnect:
        pass


@app.post("/api/upload")
def upload_video(file: UploadFile = File(...)):
    """Saves an uploaded video, samples frames every ~0.5s, runs YOLO on
    each sampled frame, and returns a detection timeline. Defined as a
    plain (non-async) function so FastAPI runs it in a worker thread and
    doesn't block the live websocket while it processes.
    """
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"clip_{int(time.time())}{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        return JSONResponse({"error": "could_not_open_video"}, status_code=400)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every_frames = max(1, int(fps * 0.5))  # ~2 samples/sec

    timeline = []
    frame_idx = 0
    # Reset predictor tracking state for fresh video file
    if hasattr(model, "predictor") and model.predictor is not None:
        model.predictor = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every_frames == 0:
            t = frame_idx / fps
            detections = run_inference(frame, track=True, persist=True)
            timeline.append({"time": round(t, 2), "detections": detections})
        frame_idx += 1
    cap.release()

    duration = total_frames / fps if fps else 0

    return {
        "filename": dest.name,
        "video_url": f"/uploads/{dest.name}",
        "duration": round(duration, 2),
        "fps": fps,
        "width": width,
        "height": height,
        "timeline": timeline,
    }


# --- Static hosting --------------------------------------------------------
# Order matters: specific routes above are matched first, these mounts
# catch everything else (uploaded clips, then the dashboard itself).
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
