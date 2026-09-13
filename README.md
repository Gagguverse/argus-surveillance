# ARGUS — AI Surveillance Intelligence System

**Built as a prototype for ONE HACK: 8 Hours AI & Web3 Hackathon.**

---

## 1. Project Overview

ARGUS is an AI-assisted aerial/drone surveillance video-analysis system that processes recorded thermal/low-light footage representing a simulated drone camera feed. It combines deterministic computer-vision pipelines (YOLOv8 + BoT-SORT tracking) with a natural-language intelligence layer (Groq AI — `openai/gpt-oss-20b`) to detect, track, and explain suspicious activity in a defined perimeter.

The system is designed for the hackathon problem: **AI-Based Border Surveillance Using Drones** — processing thermal and/or low-light video to detect and flag suspicious human movement patterns while distinguishing them from non-threat activity.

---

## 2. Problem Being Solved

Border and perimeter surveillance using drone-mounted thermal/low-light cameras presents distinct challenges compared to standard RGB video:

- **Thermal blooming** — heat signatures bleed across pixels, obscuring edges
- **Low resolution** — thermal sensors typically offer fewer pixels than optical cameras
- **No RGB colour/texture cues** — classification relies on shape and thermal contrast alone
- **Partial occlusion** — foliage, terrain, and atmospheric effects fragment detections
- **False positives** — animals, vehicles, and environmental heat sources trigger alerts

ARGUS addresses these through a layered pipeline: detection → persistent tracking → movement-vector analysis → deterministic suspicion scoring → watchlist filtering → AI-powered natural-language explanation. The computer-vision outputs remain the source of truth; the LLM only interprets and summarizes available telemetry.

---

## 3. Key Features (Implemented)

| Feature | Status | Notes |
|---------|--------|-------|
| **Thermal/low-light video support** | ✅ | Accepts any uploaded video file; no thermal-specific preprocessing |
| **YOLO object detection** | ✅ | YOLOv8n (nano, 80 COCO classes) |
| **Persistent multi-object tracking** | ✅ | BoT-SORT via `model.track(persist=True)` |
| **Movement vectors / trajectories** | ✅ | Bread-crumb trails drawn on overlay; velocity calculated per track |
| **Human detection** | ✅ | COCO `person` class |
| **Movement-pattern analysis** | ⚠️ Partial | Velocity-based only: HIGH (>15 px/frame), MODERATE (>4), STATIONARY. No gait/crouch/group classifiers |
| **Human vs non-human classification** | ✅ | Watchlist filters (person, backpack, suitcase, knife, cell phone, laptop) |
| **Threat/suspicion score** | ✅ | Deterministic 0–100 index per frame |
| **Severity levels** | ✅ | NOMINAL / ELEVATED / CRITICAL |
| **Bounding-box overlay** | ✅ | Canvas overlay with corner reticles, labels, confidence, track IDs |
| **Alert overlay** | ✅ | Security timeline with thumbnails, timestamps, track IDs |
| **Event logging** | ✅ | Rising-edge alerts for watchlisted classes |
| **Automatic snapshots** | ✅ | Thumbnail captured at alert time, viewable in lightbox |
| **Session/report generation** | ✅ | One-click HTML export with AI briefing, charts, event table |
| **AI Intelligence Panel** | ✅ | Real-time assessment, reason, recommended action |
| **Groq AI integration** | ✅ | Ultra-fast inference via Groq API (falls back to heuristic rules if no key) |
| **Natural-language surveillance briefing/chat** | ✅ | Copilot chat + session summary endpoint |

---

## 4. System Architecture

```
Thermal/Low-light / Simulated Drone Footage (uploaded video)
                    ↓
            Object Detection (YOLOv8n)
                    ↓
         Multi-Object Tracking (BoT-SORT, persistent)
                    ↓
        Movement Analysis (velocity, dwell, trails)
                    ↓
     Threat/Suspicion Scoring (deterministic, 0–100)
                    ↓
        Alerts + Events + Snapshots (rising-edge)
                    ↓
       Groq AI Intelligence Layer (interpretation)
                    ↓
      Operator-Facing Briefing / Chat (Copilot)
```

**Deterministic computer-vision outputs (detections, tracks, scores) remain the source of truth.** Groq receives structured telemetry and returns natural-language assessments, explanations, summaries, and answers to operator questions. It does not replace detection/tracking.

---

## 5. Thermal/Low-Light Challenges

| Challenge | How ARGUS Addresses It (Prototype) |
|-----------|-------------------------------------|
| Thermal blooming | No specific de-blooming; relies on YOLO's robustness to noisy edges |
| Low resolution | YOLOv8n runs at native resolution; boxes normalized for any display size |
| No RGB cues | Uses shape + confidence only; watchlist filters reduce false positives |
| Partial occlusion | BoT-SORT persists tracks across brief occlusions (`persist=True`) |
| False positives (animals/vehicles/heat) | Watchlist excludes `dog`, `car`, `bicycle` by default; they add only +5 to threat score |

**The system does NOT completely solve these challenges.** It demonstrates a functional pipeline that can be extended with thermal-specific models, domain adaptation, and richer movement classifiers.

---

## 6. Movement Patterns

**Currently implemented:** velocity-based classification only.

| Level | Velocity Threshold (px/frame) | Meaning |
|-------|-------------------------------|---------|
| HIGH | > 15 | Fast movement |
| MODERATE | > 4 | Walking-speed movement |
| STATIONARY | ≤ 4 | Dwelling / loitering |

**Prototype limitation / planned enhancement:** The system does **not** yet classify semantic patterns such as:
- Normal walking vs. running vs. sprinting
- Group/formation movement
- Crouching / crawling / prone
- Fence-line following / perimeter probing

These would require pose estimation, temporal sequence models, or thermal-specific training data.

---

## 7. Non-Threat Activity

Animals (`dog`), vehicles (`car`, `bicycle`), and other COCO classes are detected and drawn but **excluded from the default watchlist**. They:

- Appear in the detection overlay (teal boxes)
- Contribute minimally to threat score (+5 per detection)
- Do **not** trigger security-timeline alerts unless manually added to the watchlist
- Are counted in the class-breakdown statistics and AI briefing context

This mimics a real operator workflow: the system surfaces everything, the operator decides what constitutes a threat via the watchlist.

---

## 8. Dataset / Footage

The hackathon permits publicly available thermal/low-light datasets or footage. The prototype accepts any uploaded video file (MP4, WebM, MOV) as a simulated drone feed.

**Dataset used:** [ADD ACTUAL DATASET NAME]  
**Source:** [ADD SOURCE LINK]  
**Limitations:** [ADD KNOWN LIMITATIONS — e.g., resolution, frame rate, lack of ground-truth tracks, limited thermal variety]

> ⚠️ Replace the placeholders above with the actual dataset you evaluated on before submission.

---

## 9. AI — Groq Intelligence Layer

**Powered by Groq Cloud API** (`openai/gpt-oss-20b` fast model, integration present in `server.py`).

- **Ultra-fast sub-second LLM inference** — optimized for real-time surveillance operations
- Surveillance telemetry (detections, tracks, threat score, alerts, session context) is provided as structured context to the model
- The AI can:
  - Explain *why* a specific event was flagged (`/api/ai/analyze-event`)
  - Explain the current threat score (`/api/ai/explain-threat`)
  - Generate a full-session intelligence briefing (`/api/ai/session-summary`)
  - Answer operator questions in natural language (`/api/ai/copilot`)
- **It does NOT replace** the computer-vision detection/tracking system
- **Final operational decisions remain with the human operator**
- Falls back to deterministic heuristic rules when no `GROQ_API_KEY` is configured

## Drone Simulation Context

Live border/drone camera feeds are **not available** for the hackathon environment. ARGUS treats uploaded recorded footage (thermal, low-light, or optical) as the **simulated drone-camera input**. The analysis pipeline is identical to what would be applied to a live drone feed — only the ingestion mechanism differs.

ARGUS does **not** claim to:
- Control or autonomously pilot a drone
- Make autonomous military targeting decisions
- Operate on classified border/military infrastructure

Human operator review is required for all alerts. ARGUS is a decision-support tool.

## CV Pipeline

```
DETECTION (YOLOv8n) → TRACK ID (BoT-SORT) → MOVEMENT PATTERN (Heuristic) → THREAT SCORE (Deterministic) → ALERT
```

- **Detection**: YOLOv8n (nano) — 80 COCO object classes including person, vehicle, animal, weapon
- **Tracking**: BoT-SORT persistent tracking — assigns stable Track IDs across frames
- **Movement Classification** *(prototype heuristic)*: Based on pixel velocity from trail history and bounding-box aspect ratio. Classifies: Normal Walking, Fast Movement, Group Movement (≥3 persons), Low-Profile/Crouching-like [heuristic]. **Not** a trained pose model — YOLOv8n does not perform pose estimation.
- **Entity Classification**: HUMAN / ANIMAL / VEHICLE / OBJECT — derived from YOLO class labels. Non-human detections do not receive elevated human threat weighting.
- **Threat Score**: Deterministic rule-based engine (0–100). Animals and vehicles receive reduced weight vs. suspicious human movement. Low-profile movement adds a bonus.
- **Groq AI**: Natural-language explanation layer only. Does NOT own detection, tracking, or authoritative scoring. Explains telemetry, summarizes events, answers operator questions.

## Thermal / Low-Light Challenges

ARGUS is designed to handle footage from sensors that exhibit:
- Low-resolution thermal imagery
- Thermal blooming (blurred heat boundaries)
- Absence of RGB color/texture information
- Partial occlusion
- Temporal ambiguity between frames

Persistent multi-frame tracking (BoT-SORT) provides additional context, but does **not completely solve** these challenges. See the "Sensor Context & Limitations" panel in the dashboard sidebar.

---

## 10. Drone Relevance

ARGUS is the **software analysis layer** for drone-based surveillance.

- The hackathon permits public thermal/low-light footage instead of requiring live drone feeds
- Recorded aerial/thermal footage is uploaded and processed as a **simulated drone camera input**
- ARGUS does **not** control or operate a physical drone; no MAVLink, no flight stack, no hardware integration

---

## 11. Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Computer Vision | OpenCV, Ultralytics YOLOv8, PyTorch |
| Tracking | BoT-SORT (built into Ultralytics `model.track`) |
| AI Intelligence | Groq Cloud API (`openai/gpt-oss-20b` via REST API) |
| Frontend | Single-file HTML/CSS/JavaScript (no framework) |
| Real-time | WebSocket (live camera mode) |
| Video Processing | OpenCV `VideoCapture` (frame sampling ~2 fps) |
| Dependencies | `fastapi==0.115.0`, `uvicorn[standard]==0.30.6`, `opencv-python-headless==4.10.0.84`, `ultralytics==8.2.103`, `python-multipart==0.0.9`, `websockets==13.1` |

---

## 12. Project Structure

```
argus/
├── server.py              # FastAPI backend + YOLOv8 inference + Groq AI endpoints
├── requirements.txt       # Python dependencies
├── frontend/
│   └── index.html         # Full dashboard UI (HTML + CSS + JS, single file)
├── uploads/               # Saved clips (created at runtime, git-ignored)
├── yolov8n.pt             # YOLO weights (downloaded on first run, git-ignored)
├── .gitignore
└── README.md
```

---

## 13. Installation

### Windows (PowerShell)

```powershell
git clone https://github.com/Gagguverse/argus-surveillance.git
cd argus-surveillance
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux / macOS

```bash
git clone https://github.com/Gagguverse/argus-surveillance.git
cd argus-surveillance
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

First run downloads PyTorch and YOLOv8n weights (~6 MB) — do this on working Wi-Fi before the demo. The model is cached locally and works offline afterwards.

---

## 14. Running the Demo

```bash
python server.py
```

Open **http://localhost:8000** in a browser. That's the entire dashboard.

---

## 15. Demo Flow (Judge-Friendly)

1. **Input footage** — Click "FOOTAGE PLAYBACK" tab → upload a thermal/low-light video clip
2. **Detection** — YOLOv8n runs on sampled frames (~2 fps), draws boxes in real time
3. **Tracking** — BoT-SORT assigns persistent track IDs across frames
4. **Movement analysis** — Velocity computed from track history; trails drawn
5. **Suspicion score** — Deterministic 0–100 index updates per frame (watchlist-weighted)
6. **Alert/event** — Rising-edge watchlist hits create timeline entries with thumbnails
7. **AI explanation/briefing** — AI Intelligence panel updates automatically; open Copilot to ask questions

**Live camera mode** also available for real-time webcam demo (optical, not thermal).

---

## 16. Limitations & Future Work

- **Pretrained YOLOv8n** is trained on COCO (RGB, daylight); not optimized for thermal imagery — expect lower AP on thermal footage
- **Limited public thermal datasets** — prototype evaluated on [ADD DATASET]; generalization unproven
- **Movement-pattern classification** — only velocity-based (HIGH/MODERATE/STATIONARY); no gait, pose, or group semantics
- **No live drone hardware/feed** — hackathon prototype uses uploaded files only
- **Thermal blooming / low-resolution challenges** — not explicitly mitigated beyond tracking persistence
- **Single-class tracking** — BoT-SORT may swap IDs on prolonged occlusion
- **Groq API key required** for full AI features; heuristic fallback is rule-based only

---

## 17. Hackathon Context

Built as a prototype for **ONE HACK: 8 Hours AI & Web3 Hackathon**.

This is a **technical demonstration** of a surveillance-analysis pipeline combining deterministic CV with an LLM intelligence layer. It does not claim:
- Military deployment readiness
- Real-world battlefield validation
- Superior performance over dedicated thermal models
- Complete solution to thermal/low-light surveillance challenges

---

## License

Prototype code for hackathon evaluation. No warranty.