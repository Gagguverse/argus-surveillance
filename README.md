# ARGUS — AI Surveillance Dashboard

Local demo for ONE HACK. Two modes in one dashboard:
- **Live camera** — streams your webcam to the backend over a WebSocket, runs YOLOv8 on each frame, draws boxes in real time.
- **Upload footage** — upload a video file, backend samples it (~2 frames/sec) and returns a full detection timeline synced to playback.

No terminal is ever shown to judges — they only see the browser tab.

## 1. Install (one time)

```bash
cd argus
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

First install downloads PyTorch + the YOLOv8n weights, so do this **before** the demo, on working wifi. `yolov8n.pt` is the small/fast variant — good enough for real-time CPU inference on a laptop.

## 2. Run

```bash
python server.py
```

You'll see `[ARGUS] Model ready — 80 object classes available.` in the terminal — that's expected and normal, it just shouldn't be on screen during the actual demo.

Open **http://localhost:8000** in a browser. That's the whole dashboard — nothing else to open.

## 3. Demo tips

- **Live tab**: click "Start camera", allow camera permission. Bounding boxes and the alert log update as the model runs on your feed. Latency (ms per frame) shows bottom-right.
- **Upload tab**: pick a short clip (under ~30s is safest for a live demo — analysis time scales with video length). Once it's processed, the video plays with synced boxes.
- **Watchlist** (right sidebar): toggle which object classes count as alerts. Defaults are person, backpack, suitcase, knife, cell phone. Anything not checked still gets detected and drawn, just doesn't hit the alert log.
- If the camera feed looks flipped/weird: that's actually correct — this shows the true camera orientation like a real security camera, not a mirrored selfie view.

## Project structure

```
argus/
  server.py          FastAPI backend + YOLOv8 inference
  requirements.txt
  frontend/
    index.html        the whole dashboard UI (single file)
  uploads/             saved clips land here (created automatically)
```

## Troubleshooting

- **"Could not access camera"** — browser needs `localhost` (not a raw IP) or HTTPS for camera permission; `http://localhost:8000` works fine.
- **Model download stuck** — needs internet the first run only; after that `yolov8n.pt` is cached locally and works offline.
- **Slow inference on live tab** — expected on CPU-only laptops; the 600ms capture interval is tuned for that. Lower `CAPTURE_INTERVAL_MS` in `frontend/index.html` if your machine is faster, raise it if it's choppy.
