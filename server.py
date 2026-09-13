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
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from pydantic import BaseModel, ConfigDict

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Groq API Configuration (Sole AI Intelligence Provider)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = os.getenv(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions"
).strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()

# API Error state tracking for truthful fallback messages
last_api_error: Optional[str] = None


def clean_copilot_response(content: str) -> str:
    """Sanitize any model thinking, chain-of-thought, or drafting leaks."""
    if not content:
        return ""
    # Strip <think>...</think>
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>", 1)[1].strip()
    elif "</think>" in content:
        content = content.rsplit("</think>", 1)[-1].strip()
    # Strip "Here's a thinking process:"
    if "Here's a thinking process:" in content:
        if "Draft:" in content:
            content = content.split("Draft:", 1)[1].strip()
        elif "Draft Response:" in content:
            content = content.split("Draft Response:", 1)[1].strip()
        elif "\n\n" in content:
            content = content.rsplit("\n\n", 1)[-1].strip()
    # Strip "Thinking Process:"
    if "Thinking Process:" in content:
        if "\n\n" in content:
            content = content.rsplit("\n\n", 1)[-1].strip()
    # Strip "Possible response:" prefix
    if content.startswith("Possible response:"):
        content = content.split("Possible response:", 1)[1].strip()
        if content.startswith('"') and content.endswith('"'):
            content = content[1:-1].strip()
    # Strip "We need to..." drafting leaks
    if content.startswith("We need to") or content.startswith("The user is asking") or content.startswith("The user wants"):
        if "\n\n" in content:
            content = content.split("\n\n", 1)[1].strip()
        elif ". " in content:
            content = content.split(". ", 1)[1].strip()
    return content.strip()


def query_groq(messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 350, timeout: int = 12) -> Optional[str]:
    """Query the Groq API as the primary fast LLM provider without exposing keys."""
    global last_api_error
    if not GROQ_API_KEY:
        last_api_error = "no_key"
        return None

    print("[ARGUS AI] Groq request started")
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        print(f"[ARGUS AI] Groq HTTP status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                msg = choices[0]["message"]
                content = msg.get("content", "").strip()
                if not content and msg.get("reasoning"):
                    content = msg.get("reasoning", "").strip()
                if content:
                    print("[ARGUS AI] Groq response SUCCESS")
                    last_api_error = None
                    return clean_copilot_response(content)

        # Log sanitized error message without credentials
        try:
            err_json = resp.json()
            err_msg = err_json.get("error", {}).get("message", resp.text[:150])
        except Exception:
            err_msg = resp.text[:150]
        safe_err = str(err_msg).replace(GROQ_API_KEY, "[REDACTED]") if GROQ_API_KEY else str(err_msg)
        print(f"[ARGUS AI] Groq request failed: HTTP {resp.status_code} - {safe_err}")
        last_api_error = f"http_{resp.status_code}"

        # If rate limited (429), retry once after 1 second backoff
        if resp.status_code == 429:
            time.sleep(1.0)
            resp2 = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
            print(f"[ARGUS AI] Groq retry HTTP status: {resp2.status_code}")
            if resp2.status_code == 200:
                data2 = resp2.json()
                choices2 = data2.get("choices", [])
                if choices2 and "message" in choices2[0]:
                    msg2 = choices2[0]["message"]
                    content2 = msg2.get("content", "").strip()
                    if not content2 and msg2.get("reasoning"):
                        content2 = msg2.get("reasoning", "").strip()
                    if content2:
                        print("[ARGUS AI] Groq retry response SUCCESS")
                        last_api_error = None
                        return clean_copilot_response(content2)

    except requests.exceptions.Timeout:
        print("[ARGUS AI] Groq request timed out")
        last_api_error = "timeout"
    except Exception as e:
        safe_exc = str(e).replace(GROQ_API_KEY, "[REDACTED]") if GROQ_API_KEY else str(e)
        print(f"[ARGUS AI] Groq request failed: {safe_exc}")
        last_api_error = "error"
    return None


app = FastAPI(title="ARGUS Surveillance API with Groq AI")

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

# Patch Ultralytics BoT-SORT GMC (Global Motion Compensation) to prevent
# cv2.calcOpticalFlowPyrLK assertion crash when video/camera resolution changes:
# 'prevPyr[level * lvlStep1].size() == nextPyr[level * lvlStep2].size()'
try:
    from ultralytics.trackers.utils.gmc import GMC

    _orig_applySparseOptFlow = GMC.applySparseOptFlow

    def _safe_applySparseOptFlow(self, raw_frame: np.ndarray) -> np.ndarray:
        h, w = raw_frame.shape[:2]
        downscale = getattr(self, "downscale", 2.0)
        curr_shape = (h // int(downscale), w // int(downscale)) if downscale > 1.0 else (h, w)
        if hasattr(self, "prevFrame") and self.prevFrame is not None:
            if self.prevFrame.shape != curr_shape:
                self.reset_params()
        try:
            return _orig_applySparseOptFlow(self, raw_frame)
        except cv2.error as e:
            if "size()" in str(e) or "calcOpticalFlowPyrLK" in str(e):
                self.reset_params()
                return np.eye(2, 3)
            raise

    GMC.applySparseOptFlow = _safe_applySparseOptFlow
    print("[ARGUS] BoT-SORT GMC multi-resolution patch active.")
except Exception as patch_err:
    print(f"[ARGUS] Warning: GMC patch not applied ({patch_err})")

DEFAULT_WATCHLIST = sorted(
    {"person", "backpack", "suitcase", "knife", "cell phone", "laptop"}
)


def classify_activity(
    label: str,
    velocity: float,
    box: Optional[List[float]] = None,
    all_detections: Optional[List[Dict[str, Any]]] = None,
    track_history: Optional[List[Dict[str, Any]]] = None,
    height_history: Optional[List[float]] = None,
) -> str:
    """Classifies surveillance entity activity:
    - STATIONARY
    - NORMAL WALKING
    - FAST MOVEMENT / RUNNING
    - CYCLING
    - GROUP MOVEMENT
    - LOW-PROFILE (HEURISTIC)
    - OTHER
    """
    if label != "person" and label != "bicycle":
        if velocity < 3.0:
            return "STATIONARY"
        elif velocity >= 14.0:
            return "FAST MOVEMENT / RUNNING"
        elif velocity >= 3.0:
            return "MOVING"
        return "OTHER"

    # Minimal velocity check -> STATIONARY
    if velocity < 3.0:
        return "STATIONARY"

    # 1. CYCLING:
    # Requires person + bicycle spatial association + sustained movement
    if all_detections and box:
        px1, py1, px2, py2 = box
        px_c, py_c = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        bicycles = [d for d in all_detections if d.get("label") == "bicycle"]
        for b in bicycles:
            bbox = b.get("box")
            if bbox:
                bx1, by1, bx2, by2 = bbox
                bx_c, by_c = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                dist = math.hypot(px_c - bx_c, py_c - by_c)
                overlap_x = max(0.0, min(px2, bx2) - max(px1, bx1))
                spatial_evidence = (dist < 0.28) or (overlap_x > 0 and py2 >= by1 - 0.05 and abs(px_c - bx_c) < 0.20)
                if spatial_evidence and velocity >= 3.5:
                    return "CYCLING"

    # 2. LOW-PROFILE (HEURISTIC):
    # Clearly labelled prototype heuristic using aspect ratio (w/h >= 0.85) or relative height drop >= 30%
    if box and label == "person":
        bw = box[2] - box[0]
        bh = box[3] - box[1]
        aspect_ratio = (bw / bh) if bh > 0 else 0
        is_squat = aspect_ratio >= 0.85
        has_height_drop = False
        if height_history and len(height_history) >= 2:
            max_h = max(height_history)
            if max_h > 0 and bh <= 0.70 * max_h:
                has_height_drop = True

        if (is_squat or has_height_drop) and velocity < 14.0:
            return "LOW-PROFILE (HEURISTIC)"

    # 3. FAST MOVEMENT / RUNNING
    if velocity >= 14.0:
        return "FAST MOVEMENT / RUNNING"

    # 4. GROUP MOVEMENT:
    # Multiple nearby persistent human Track IDs moving coherently
    if all_detections and box and label == "person":
        px1, py1, px2, py2 = box
        px_c, py_c = (px1 + px2) / 2.0, (py1 + py2) / 2.0
        other_moving = 0
        for d in all_detections:
            if d.get("label") == "person" and d.get("box") != box:
                obox = d.get("box")
                if obox:
                    ox1, oy1, ox2, oy2 = obox
                    ox_c, oy_c = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
                    dist = math.hypot(px_c - ox_c, py_c - oy_c)
                    other_vel = d.get("velocity", 0.0)
                    if dist < 0.35 and other_vel >= 3.0:
                        other_moving += 1
        if other_moving >= 1:
            return "GROUP MOVEMENT"

    # 5. NORMAL WALKING
    if label == "person" and 3.0 <= velocity < 14.0:
        return "NORMAL WALKING"

    return "OTHER"


TRACKER_CONFIG_PATH = str(Path(__file__).parent / "surveillance_botsort.yaml")


def run_inference(
    frame: np.ndarray,
    conf: float = 0.25,
    track: bool = False,
    persist: bool = True,
) -> List[Dict[str, Any]]:
    """Run YOLO on a single BGR frame, return normalized detections with optional tracking IDs."""
    if track:
        tracker_file = TRACKER_CONFIG_PATH if os.path.exists(TRACKER_CONFIG_PATH) else "botsort.yaml"
        results = model.track(
            frame,
            conf=conf,
            tracker=tracker_file,
            persist=persist,
            verbose=False,
        )
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


# --- Groq AI Intelligence Endpoints ---------------------------------------

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


# -----------------------------------------------------------------------
# Server-side Cooldown and Caching for Explain-Threat and Analyze-Event
# -----------------------------------------------------------------------
EXPLAIN_THREAT_COOLDOWN = 8.0  # seconds

class ExplainThreatCache:
    def __init__(self):
        self.last_time: float = 0.0
        self.cached_explanation: Optional[str] = None
        self.last_score: int = -1
        self.last_level: str = ""
        self.last_classes: List[str] = []
        self.last_total_objects: int = 0
        self.last_has_critical: bool = False
        self.last_movement_level: str = ""

    def is_similar(self, req: ThreatExplainRequest) -> bool:
        """Determines if the state has NOT changed meaningfully."""
        if not self.cached_explanation:
            return False
        # Meaningful changes that invalidate cache even within cooldown:
        if req.threat_level != self.last_level:
            return False
        if req.has_critical != self.last_has_critical:
            return False
        if set(req.active_classes) != set(self.last_classes):
            return False
        if req.movement_level != self.last_movement_level:
            return False
        if abs(req.threat_score - self.last_score) > 10:
            return False
        if abs(req.total_objects - self.last_total_objects) >= 3:
            return False
        return True

    def get(self, req: ThreatExplainRequest) -> Optional[str]:
        now = time.time()
        if (now - self.last_time < EXPLAIN_THREAT_COOLDOWN) and self.is_similar(req):
            return self.cached_explanation
        return None

    def set(self, req: ThreatExplainRequest, explanation: str):
        self.last_time = time.time()
        self.cached_explanation = explanation
        self.last_score = req.threat_score
        self.last_level = req.threat_level
        self.last_classes = list(req.active_classes)
        self.last_total_objects = req.total_objects
        self.last_has_critical = req.has_critical
        self.last_movement_level = req.movement_level

explain_cache = ExplainThreatCache()

# Deduplication cache for analyze-event: key -> (timestamp, result_dict)
EVENT_ANALYSIS_COOLDOWN = 10.0
event_analysis_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


class SessionSummaryRequest(BaseModel):
    duration: float = 0.0
    total_alerts: int = 0
    total_tracked: int = 0
    highest_threat: int = 0
    detected_classes: Dict[str, int] = {}
    key_events: List[Dict[str, Any]] = []


class CopilotRequest(BaseModel):
    question: str
    session_context: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    sessionContext: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None
    elapsed: Optional[str] = None
    duration: Optional[str] = None
    alerts: Optional[Any] = None
    total_alerts: Optional[Any] = None
    objects: Optional[Any] = None
    total_objects: Optional[Any] = None
    threat_score: Optional[Any] = None
    threat_level: Optional[str] = None
    classes: Optional[Dict[str, Any]] = None
    detected_classes: Optional[Dict[str, Any]] = None
    active_tracks: Optional[List[Dict[str, Any]]] = None
    activeTracks: Optional[List[Dict[str, Any]]] = None
    recent_alerts: Optional[List[Dict[str, Any]]] = None
    recentAlerts: Optional[List[Dict[str, Any]]] = None

    model_config = ConfigDict(extra="allow")


@app.get("/api/ai/status")
def ai_status():
    """Returns AI intelligence status and active engine information."""
    return {
        "configured": bool(GROQ_API_KEY),
        "engine": "Groq" if GROQ_API_KEY else "ARGUS Threat Intelligence Engine",
        "model": GROQ_MODEL if GROQ_API_KEY else "Deterministic Rule-Based Analyst",
        "primary": "Groq" if GROQ_API_KEY else "Deterministic Rule-Based Analyst",
        "fallback": "Telemetry-only",
    }


@app.post("/api/ai/analyze-event")
def analyze_event(req: EventAnalysisRequest):
    """Analyze a single security event or alert using Groq, with deduplication."""
    now = time.time()
    event_key = f"{req.target_class}_{req.track_id if req.track_id is not None else 'notrack'}_{req.severity}"
    if event_key in event_analysis_cache:
        cached_time, cached_result = event_analysis_cache[event_key]
        if now - cached_time < EVENT_ANALYSIS_COOLDOWN:
            print(f"[ARGUS AI] Analyze-event cache HIT for {event_key}")
            return cached_result

    print(f"[ARGUS AI] Analyze-event cache MISS for {event_key}")

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

    response_text = query_groq(messages, temperature=0.1, max_tokens=250, timeout=12)
    result = None
    if response_text:
        try:
            # Strip markdown if model included it
            clean = response_text.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1]
                if clean.endswith("```"):
                    clean = clean.rsplit("```", 1)[0]
                clean = clean.strip()
            result = json.loads(clean)
        except Exception:
            pass

    if not result:
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

        result = {
            "assessment": assessment,
            "severity": severity,
            "reason": reason,
            "recommended_action": recommended_action,
        }

    # Store in deduplication cache
    event_analysis_cache[event_key] = (now, result)
    if len(event_analysis_cache) > 50:
        cutoff = now - 60.0
        for k in list(event_analysis_cache.keys()):
            if event_analysis_cache[k][0] < cutoff:
                del event_analysis_cache[k]

    return result


@app.post("/api/ai/explain-threat")
def explain_threat(req: ThreatExplainRequest):
    """Explains why the current ARGUS threat score is at its current level, with server-side caching."""
    cached = explain_cache.get(req)
    if cached:
        print(f"[ARGUS AI] Explain-threat cache HIT")
        return {"explanation": cached}

    print(f"[ARGUS AI] Explain-threat cache MISS")

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

    ai_exp = query_groq(messages, temperature=0.2, max_tokens=150, timeout=12)
    if ai_exp:
        explanation = ai_exp.strip()
    else:
        # Heuristic fallback
        if req.has_critical:
            explanation = f"Threat level is critical ({req.threat_score}/100) due to confirmed weapon detection in the monitored sector."
        elif req.threat_score >= 50:
            explanation = f"Threat score elevated to {req.threat_score}/100 due to multiple active targets ({classes_str}) with {req.movement_level.lower()} activity."
        elif req.threat_score >= 25:
            explanation = f"Threat index at {req.threat_score}/100 indicating normal presence of {classes_str} under observation."
        else:
            explanation = "Threat status nominal (0/100). No security threats or watchlisted anomalies detected in sector."

    explain_cache.set(req, explanation)
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

    response_text = query_groq(messages, temperature=0.15, max_tokens=350, timeout=15)
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


def clean_copilot_broad_summary(text: str, fallback_dashboard: str) -> str:
    """Ensure broad Copilot replies conform strictly to the 6-bullet dashboard format."""
    if not text:
        return fallback_dashboard
    lower = text.lower()
    idx = lower.find("session analysis")
    if idx != -1:
        header_end = text.find("\n", idx)
        if header_end != -1:
            body = text[header_end + 1:]
            lines = []
            for line in body.split("\n"):
                s = line.strip()
                if not s:
                    continue
                # Strip markdown bolding on bullet label if present
                for h in ("Objects detected:", "Active tracks:", "Movement:", "Overall threat:", "Security alerts:", "Assessment:"):
                    s = s.replace(f"**{h}**", h).replace(f"**{h.rstrip(':')}**:", h)
                # Normalize bullets
                if s.startswith(("- ", "* ", "• ")):
                    s = "• " + s[2:].strip()
                elif s.startswith(("-", "*", "•")):
                    s = "• " + s[1:].strip()
                lines.append(s)
            if len(lines) >= 4:
                return "Session analysis:\n" + "\n".join(lines)
    # If the LLM didn't include the header but outputted bullet points
    lines = []
    for line in text.strip().split("\n"):
        s = line.strip()
        if not s:
            continue
        for h in ("Objects detected:", "Active tracks:", "Movement:", "Overall threat:", "Security alerts:", "Assessment:"):
            s = s.replace(f"**{h}**", h).replace(f"**{h.rstrip(':')}**:", h)
        if s.startswith(("- ", "* ", "• ")):
            lines.append("• " + s[2:].strip())
        elif s.startswith(("-", "*", "•")):
            lines.append("• " + s[1:].strip())
    if len(lines) >= 4:
        return "Session analysis:\n" + "\n".join(lines)
    return fallback_dashboard


@app.post("/api/ai/copilot")
def copilot_chat(req: CopilotRequest):
    """Answer user questions about the current surveillance session with grounded context."""
    # Debug logging — never logs the API key
    print(f"[Copilot] Received question: {req.question[:150]}")

    # Merge telemetry context from all possible sources (nested and top-level)
    raw_ctx = {}
    if req.session_context and isinstance(req.session_context, dict):
        raw_ctx.update(req.session_context)
    if req.context and isinstance(req.context, dict):
        raw_ctx.update(req.context)
    if req.sessionContext and isinstance(req.sessionContext, dict):
        raw_ctx.update(req.sessionContext)
    if req.telemetry and isinstance(req.telemetry, dict):
        raw_ctx.update(req.telemetry)

    # Safe fallback if telemetry context is missing entirely
    has_telemetry = bool(
        raw_ctx or req.active_tracks or req.activeTracks or req.classes
        or req.detected_classes or (req.objects is not None)
        or (req.threat_score is not None)
    )
    if not has_telemetry:
        q_lower = req.question.lower()
        if any(k in q_lower for k in ["who are you", "what is your name", "identify yourself", "what can you do"]):
            return {
                "answer": "I am ARGUS Copilot, the AI intelligence assistant for ARGUS. I interpret live surveillance telemetry from the YOLOv8 and BoT-SORT computer vision pipeline.",
                "provider": "ARGUS Core"
            }
        return {
            "answer": "Telemetry context is currently unavailable from the surveillance client. Please ensure the live camera feed or video file is active to provide real-time tracking, object, and threat telemetry.",
            "provider": "ARGUS Core"
        }

    # Top-level fields take precedence or fill in missing fields:
    duration = req.elapsed or req.duration or raw_ctx.get("elapsed") or raw_ctx.get("duration") or "00:00"

    alerts_val = req.alerts if req.alerts is not None else req.total_alerts if req.total_alerts is not None else raw_ctx.get("alerts", raw_ctx.get("total_alerts", 0))
    try:
        alerts = int(alerts_val)
    except (ValueError, TypeError):
        alerts = 0

    active_tracks = req.active_tracks or req.activeTracks or raw_ctx.get("active_tracks") or raw_ctx.get("activeTracks") or []
    if not isinstance(active_tracks, list):
        active_tracks = []

    raw_objects = req.objects if req.objects is not None else req.total_objects if req.total_objects is not None else raw_ctx.get("objects", raw_ctx.get("total_objects", 0))
    try:
        objects = int(raw_objects)
    except (ValueError, TypeError):
        objects = len(active_tracks) if active_tracks else 0

    # If objects was 0 or omitted but active_tracks has items, infer count from active_tracks
    if objects == 0 and active_tracks:
        objects = len(active_tracks)

    threat_val = req.threat_score if req.threat_score is not None else raw_ctx.get("threat_score", raw_ctx.get("threatScore", 0))
    try:
        threat = int(threat_val)
    except (ValueError, TypeError):
        threat = 0

    threat_level = req.threat_level or raw_ctx.get("threat_level") or raw_ctx.get("threatLevel")
    if not threat_level:
        if threat >= 80:
            threat_level = "CRITICAL"
        elif threat >= 40:
            threat_level = "ELEVATED"
        elif threat >= 20:
            threat_level = "NOMINAL"
        else:
            threat_level = "NOMINAL"

    classes = req.classes or req.detected_classes or raw_ctx.get("classes") or raw_ctx.get("detected_classes") or {}
    if not isinstance(classes, dict):
        classes = {}
    if not classes and active_tracks:
        for t in active_tracks:
            lbl = t.get("label", "target")
            classes[lbl] = classes.get(lbl, 0) + 1

    recent_alerts = req.recent_alerts or req.recentAlerts or raw_ctx.get("recent_alerts") or raw_ctx.get("recentAlerts") or []
    if not isinstance(recent_alerts, list):
        recent_alerts = []

    # Sanitize recent_alerts to strip any heavy base64 thumbnails/images
    clean_recent_alerts = []
    for a in recent_alerts[:8]:
        if isinstance(a, dict):
            clean_recent_alerts.append({
                k: v for k, v in a.items()
                if k not in ("thumb", "thumbnail", "image", "frame", "base64")
            })
        else:
            clean_recent_alerts.append(a)

    # Sanitize active_tracks to ensure no heavy payload leaks
    clean_active_tracks = []
    for t in active_tracks:
        if isinstance(t, dict):
            clean_active_tracks.append({
                k: v for k, v in t.items()
                if k not in ("thumb", "thumbnail", "image", "frame", "trail")
            })
        else:
            clean_active_tracks.append(t)

    # Distinguish genuine security alerts from routine informational logs
    meaningful_alerts = []
    informational_logs = []
    for a in clean_recent_alerts:
        sev = str(a.get("severity", "")).upper()
        lbl = str(a.get("label", "")).lower()
        mp = str(a.get("movementPattern", "")).upper()
        if sev in ("CRITICAL", "ELEVATED") or mp.startswith("LOW-PROFILE") or lbl in ("knife", "backpack", "suitcase"):
            meaningful_alerts.append(a)
        else:
            informational_logs.append(a)

    # Format tracks in natural language for clarity (explicitly distinguishing baseline risk contribution)
    tracks_formatted = []
    for t in clean_active_tracks:
        tid = t.get("track_id", t.get("trackId", "N/A"))
        lbl = t.get("label", "target")
        mp = t.get("movementPattern", t.get("movement_level", "NORMAL"))
        vel = t.get("velocity", 0.0)
        risk = t.get("risk_contribution", t.get("threat", 20))
        tracks_formatted.append(f"{lbl} #{tid} [Activity: {mp}, Velocity: {vel} m/s, Base Risk Contribution: {risk} pts]")
    tracks_str = "; ".join(tracks_formatted) if tracks_formatted else "None"

    classes_str = ", ".join(f"{k}: {v}" for k, v in classes.items()) if classes else "None"

    # Format dashboard summary telemetry fields
    if classes:
        parts = []
        for k, v in classes.items():
            k_clean = str(k).strip()
            if v == 1:
                parts.append(f"1 {k_clean}")
            else:
                plural = k_clean if k_clean.endswith("s") else f"{k_clean}s"
                parts.append(f"{v} {plural}")
        objs_detected_str = ", ".join(parts)
    elif objects > 0:
        objs_detected_str = f"{objects} {'person' if objects == 1 else 'persons'}"
    else:
        objs_detected_str = "0 objects detected"

    active_tracks_count_str = str(len(clean_active_tracks))

    if clean_active_tracks:
        mov_items = []
        for t in clean_active_tracks:
            tid = t.get("track_id", t.get("trackId", "N/A"))
            mp = t.get("movementPattern", t.get("movement_level", "NORMAL"))
            vel = t.get("velocity", 0.0)
            mov_items.append(f"Track #{tid} — {mp} ({vel} m/s)")
        movement_str = "; ".join(mov_items)
    else:
        movement_str = "No active tracks"

    overall_threat_str = f"{threat}/100 ({threat_level})"

    if meaningful_alerts:
        alert_items = []
        for a in meaningful_alerts:
            tid = a.get("trackId", a.get("track_id", "N/A"))
            lbl = a.get("label", "entity")
            sev = a.get("severity", "ELEVATED")
            reason = a.get("reason", "Threshold reached")
            alert_items.append(f"Track #{tid} ({lbl}) — {sev}: {reason}")
        security_alerts_str = f"{len(meaningful_alerts)} active ({'; '.join(alert_items)})"
    else:
        security_alerts_str = "No active security alerts"

    if threat_level == "CRITICAL":
        assessment_str = "CRITICAL security condition active; immediate operator intervention required."
    elif threat_level == "ELEVATED":
        assessment_str = "Elevated activity detected; surveillance operator should monitor active tracks."
    else:
        assessment_str = "Current telemetry indicates routine surveillance activity with no elevated or critical condition active."

    fallback_dashboard = (
        f"Session analysis:\n"
        f"• Objects detected: {objs_detected_str}\n"
        f"• Active tracks: {active_tracks_count_str}\n"
        f"• Movement: {movement_str}\n"
        f"• Overall threat: {overall_threat_str}\n"
        f"• Security alerts: {security_alerts_str}\n"
        f"• Assessment: {assessment_str}"
    )

    q = req.question.strip().lower()
    broad_triggers = [
        "what did you analyze", "what did you detect", "what is happening",
        "summarize the session", "session analysis", "what are you analyzing",
        "what have you analyzed", "what was analyzed", "what is detected",
        "what have you detected", "summarize", "summary of the session",
        "session summary", "overview of the session", "system overview",
        "what happened", "give me a summary", "briefing", "session status"
    ]
    is_broad = any(t in q for t in broad_triggers) or q.rstrip("?.!") in [
        "status", "overview", "what is the status", "what is the current status"
    ]

    system_prompt = f"""SYSTEM:
You are ARGUS Copilot, the AI intelligence assistant for ARGUS.

CRITICAL OPERATIONAL RULES & TELEMETRY GROUNDING:
1. THE DETERMINISTIC THREAT ENGINE IS THE ONLY SOURCE OF TRUTH:
   - The authoritative Overall Threat Score is {overall_threat_str}. This is the ONLY value that may be called the "Threat Score".
   - Per-track values are individual baseline risk contributions (e.g. 20 pts for a person), NOT threat scores. NEVER call a per-track value "the threat score".
2. NORMAL & STATIONARY ENTITIES ARE NOT SUSPICIOUS:
   - A stationary person (pattern: STATIONARY) or walking person (pattern: NORMAL WALKING) has a nominal baseline presence.
   - Do NOT describe a stationary person or walking pedestrian as suspicious, elevated, or high-threat.
3. DISTINGUISH ACTUAL SECURITY ALERTS FROM ROUTINE LOGS:
   - If there are no critical or elevated security alerts (meaningful alerts list is empty or 0), you MUST explicitly say "No active security alerts."
   - Routine detection logs (e.g. standard person presence) are informational events, NOT security violations.
   - If an alert was generated by a heuristic (e.g. LOW-PROFILE (HEURISTIC)), explicitly state that it was flagged by a visual geometry heuristic (aspect ratio / height drop), not confirmed threat or pose detection.
4. STRICT ADHERENCE TO TELEMETRY:
   - Rely STRICTLY and EXCLUSIVELY on the CURRENT SESSION TELEMETRY below. Do NOT infer, invent, or assume unmentioned weapons, hostile intent, or past security breaches.
   - If any requested metric is missing in telemetry, state that data is unavailable rather than guessing.

RESPONSE FORMAT RULES:
A. BROAD QUESTIONS (e.g. "what did you analyze?", "what did you detect?", "what is happening?", "summarize the session", "status", "overview", "what happened", "what are you analyzing"):
   You MUST return ONLY a concise dashboard-style summary with this exact structure (no preamble, no conversational filler, 4–6 bullet points using '•'):
   Session analysis:
   • Objects detected: {objs_detected_str}
   • Active tracks: {active_tracks_count_str}
   • Movement: {movement_str}
   • Overall threat: {overall_threat_str}
   • Security alerts: {security_alerts_str}
   • Assessment: {assessment_str}

B. SPECIFIC QUESTIONS (e.g. asking specifically about movement, velocity, threat score, specific track ID, cycling, heuristic alert, object counts):
   Answer that specific question DIRECTLY and factually in 1–3 concise sentences.
   Do NOT output the full "Session analysis:" block for specific questions.

CURRENT SESSION TELEMETRY:
- Session Duration: {duration}
- Objects detected: {objs_detected_str}
- Active tracks count: {active_tracks_count_str}
- Active tracks movement: {movement_str}
- Overall threat: {overall_threat_str}
- Security alerts: {security_alerts_str}
- Assessment: {assessment_str}
- Active Tracked Entities Details: {tracks_str}
- Critical/Elevated Security Alerts Details: {json.dumps(meaningful_alerts) if meaningful_alerts else "None (session clear, 0 active security alerts)"}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.question}
    ]

    # 1. Call Groq directly as sole external LLM provider
    has_key = bool(GROQ_API_KEY)
    if has_key:
        print("[Copilot] Calling Groq: YES")
        groq_reply = query_groq(messages, temperature=0.1, max_tokens=350, timeout=12)
        if groq_reply:
            print(f"[Copilot] Groq response: SUCCESS ({len(groq_reply)} chars)")
            if is_broad:
                groq_reply = clean_copilot_broad_summary(groq_reply, fallback_dashboard)
            return {"answer": groq_reply, "provider": "Groq"}
        print("[Copilot] Groq call failed/timed out. Using fallback: YES")
    else:
        print("[Copilot] Calling Groq: NO — GROQ_API_KEY not set. Using fallback: YES")

    # ---------------------------------------------------------------
    # Telemetry-only fallback — clearly labelled so the operator knows
    # Groq is not responding. Different questions produce different
    # answers by matching against the actual question text.
    # ---------------------------------------------------------------
    if is_broad:
        return {"answer": fallback_dashboard, "provider": "Groq-Fallback" if has_key else "Telemetry-only"}

    if has_key:
        if last_api_error == "timeout":
            fallback_reason = "Groq is temporarily unavailable due to an API/network timeout. Telemetry-only fallback is active."
        else:
            fallback_reason = "Groq is temporarily unavailable due to an API/network error. Telemetry-only fallback is active."
        PREFIX = f"⚠️ {fallback_reason}\n\n"
    else:
        fallback_reason = "Groq AI is currently unavailable (no GROQ_API_KEY configured)."
        PREFIX = "⚠️ Groq unavailable. Telemetry-only fallback:\n\n"

    # Identity / capability questions — check first, always unambiguous
    if any(k in q for k in ["who are you", "what is your name", "identify yourself", "are you working",
                              "are you online", "are you active", "what can you do"]):
        reply = ("I am ARGUS Copilot, the natural-language intelligence layer for ARGUS. "
                 f"{fallback_reason} "
                 f"Session status: {threat_level} ({threat}/100).")

    # Specific query: why critical if humans only / classification rules
    elif any(k in q for k in ["humans only", "only humans", "human only", "why critical"]):
        classes_str = ", ".join(f"{k}({v})" for k, v in classes.items()) if classes else "none"
        reply = (f"Humans alone do not trigger CRITICAL threat status. Under ARGUS threat scoring, "
                 f"CRITICAL (80-100/100) strictly requires a weapon detection (e.g. knife) or critical security alert. "
                 f"Unarmed persons are scored at base ~20pts (or up to ~45pts with group/crouching movement). "
                 f"Current session status: {threat_level} ({threat}/100) with detected classes: {classes_str}.")

    # Movement / tracking questions
    elif any(k in q for k in ["moved", "movement", "tracking", "pattern", "walking", "crouching",
                               "low-profile", "group", "cycling", "velocity", "suspicious movement"]):
        if active_tracks:
            try:
                top_track = max(active_tracks, key=lambda x: x.get("velocity", 0))
            except Exception:
                top_track = active_tracks[0]
            mp = top_track.get("movementPattern", top_track.get("movement_level", "UNKNOWN"))
            tid = top_track.get("track_id", top_track.get("trackId", "N/A"))
            lbl = top_track.get("label", "entity")
            vel = top_track.get("velocity", 0.0)
            risk = top_track.get("risk_contribution", top_track.get("threat", 20))

            if mp == "STATIONARY":
                note = f"Track #{tid} ({lbl}) is stationary (velocity {vel} m/s) with a nominal baseline risk contribution of {risk} pts. It is not considered suspicious."
            elif mp == "NORMAL WALKING":
                note = f"Track #{tid} ({lbl}) is walking normally (velocity {vel} m/s) with nominal baseline risk contribution of {risk} pts."
            elif mp.startswith("CYCLING"):
                note = f"Track #{tid} ({lbl}) is actively cycling (velocity {vel} m/s, consistent trajectory)."
            elif mp.startswith("LOW-PROFILE"):
                note = f"Track #{tid} ({lbl}) was flagged as {mp} based on bounding-box aspect ratio / height drop heuristic."
            else:
                note = f"Track #{tid} ({lbl}) movement pattern: {mp} (velocity {vel} m/s)."

            reply = (f"{note} Total active tracks: {len(active_tracks)}. Overall session threat score: {threat}/100 ({threat_level}).")
        else:
            reply = "No active tracks currently detected in the surveillance area."

    # Threat / risk / why critical questions
    elif any(k in q for k in ["threat", "risk", "critical", "score", "danger",
                               "why is", "why critical", "why elevated"]):
        classes_str = ", ".join(f"{k}({v})" for k, v in classes.items()) if classes else "none"
        track_summary = ""
        if active_tracks:
            track_summary = " Active tracks: " + "; ".join(
                f"{t.get('label','?')} #{t.get('track_id','?')} [{t.get('movementPattern', t.get('movement_level','?'))}]"
                for t in active_tracks[:4]
            ) + "."
        if threat_level == "NOMINAL":
            status_desc = f"Current overall threat score is {threat}/100 (NOMINAL). No weapons or critical violations are present. All detected entities are at standard baseline contribution."
        else:
            status_desc = f"Current overall threat score is {threat}/100 ({threat_level}) with detected classes: {classes_str}."
        reply = f"{status_desc}{track_summary}"

    # People / human count questions
    elif any(k in q for k in ["how many", "people", "person", "human", "tracked", "entities"]):
        person_tracks = [t for t in active_tracks if t.get("label") == "person"]
        reply = (f"Current objects in frame: {objects}. "
                 f"Person tracks active: {len(person_tracks)}. "
                 f"Total tracked entities: {len(active_tracks)}. "
                 f"Critical/Elevated security alerts: {len(meaningful_alerts)}.")

    # Why flagged / alert explanation
    elif "why" in q and any(k in q for k in ["flagged", "alert", "detected", "triggered", "status"]):
        if meaningful_alerts:
            latest = meaningful_alerts[0]
            reply = (f"Latest security alert: {latest.get('label','entity')} Track #{latest.get('trackId','N/A')} "
                     f"flagged with severity {latest.get('severity','ELEVATED')}. Reason: {latest.get('reason','Security threshold reached')}.")
        elif clean_recent_alerts:
            reply = "No critical or elevated security alerts have been triggered. The log contains only routine informational detections (e.g. standard person presence)."
        else:
            reply = "No security alerts have been logged in this session. The surveillance area is currently clear."

    # Session / briefing / what happened
    elif any(k in q for k in ["session", "briefing", "summary", "what happened", "report"]):
        classes_str = ", ".join(f"{k}: {v}" for k, v in classes.items()) if classes else "none detected"
        alert_info = f"{len(meaningful_alerts)} critical/elevated alerts" if meaningful_alerts else "0 critical/elevated alerts (clear)"
        reply = (f"SESSION BRIEFING\n"
                 f"Duration: {duration}\n"
                 f"Objects currently in frame: {objects}\n"
                 f"Active tracked entities: {len(active_tracks)}\n"
                 f"Security alerts: {alert_info}\n"
                 f"Authoritative threat score: {threat}/100 ({threat_level})\n"
                 f"Class breakdown: {classes_str}")

    # Animals / vehicles questions
    elif any(k in q for k in ["animal", "vehicle", "car", "dog", "wildlife"]):
        animal_tracks = [t for t in active_tracks if t.get("category") == "animal"]
        vehicle_tracks = [t for t in active_tracks if t.get("category") == "vehicle"]
        reply = (f"Animal tracks in frame: {len(animal_tracks)}. "
                 f"Vehicle tracks: {len(vehicle_tracks)}. "
                 f"Non-human detections receive reduced threat weights and do not trigger elevated alerts by default.")

    # Generic catch-all — still tries to be relevant to the question
    else:
        tip = "Telemetry fallback active while remote API recovers." if has_key else "For detailed analysis, configure GROQ_API_KEY to enable full Groq intelligence."
        reply = (f"Question received: '{req.question}'. "
                 f"Telemetry context: threat {threat}/100 ({threat_level}), "
                 f"{objects} objects in frame, {len(meaningful_alerts)} security alerts logged over {duration}. "
                 f"{fallback_reason} {tip}")

    return {"answer": PREFIX + reply}


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

            detections = run_inference(frame, track=True, persist=True)
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
