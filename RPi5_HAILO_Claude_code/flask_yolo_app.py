#!/usr/bin/env python3
"""Flask + Hailo YOLOv8 live detection server - optimized with decoupled pipeline.

Architecture:
  Thread 1 (Capture)   : Grabs frames from camera as fast as possible
  Thread 2 (Inference)  : Takes latest frame, runs YOLO, overlays results
  Thread 3 (Flask)      : Serves pre-encoded MJPEG to all clients

This decoupling ensures camera capture isn't blocked by inference,
and streaming isn't blocked by either.
"""

import cv2
import numpy as np
import time
import threading
import glob
import os
from flask import Flask, Response, render_template_string, jsonify, request
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams
)

# --- COCO classes ---
COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]
COLORS = np.random.default_rng(42).integers(0, 255, size=(80, 3)).tolist()

HEF_PATH = "/usr/share/hailo-models/yolov8s_h8.hef"
CONF_THRESHOLD = 0.4
VIDEO_DIR = "/home/pw/ws"

# --- Performance tuning ---
CAPTURE_WIDTH = 640       # Capture resolution
CAPTURE_HEIGHT = 480      # Capture resolution
INFER_EVERY_N = 2         # Run YOLO every N frames; reuse bbox on skipped frames
JPEG_QUALITY = 65         # JPEG quality (was 80) - lower = faster encoding
OUTPUT_MAX_WIDTH = 480    # Resize output before JPEG encoding (0 = no resize)

app = Flask(__name__)


# --- Shared state with decoupled pipeline ---
class DetectionState:
    def __init__(self):
        # Latest raw frame from camera (written by capture thread, read by inference thread)
        self._capture_lock = threading.Lock()
        self._raw_frame = None
        self._frame_seq = 0  # incremented on each new capture

        # Latest annotated output (written by inference thread, read by streaming)
        self._output_lock = threading.Lock()
        self._annotated_frame = None
        self._jpeg_buffer = None  # pre-encoded JPEG for streaming
        self._detections = []
        self._fps_capture = 0.0
        self._fps_infer = 0.0
        self._fps_stream = 0.0
        self._infer_ms = 0.0
        self._last_infer_seq = -1

        # Source control
        self.source = "camera"
        self.running = False
        self.restart_flag = False
        # Mode: "yolo" (with inference) or "passthrough" (raw camera, no YOLO)
        self.mode = "yolo"

    def put_raw_frame(self, frame):
        with self._capture_lock:
            self._raw_frame = frame
            self._frame_seq += 1

    def get_raw_frame(self):
        with self._capture_lock:
            return self._raw_frame, self._frame_seq

    def put_output(self, annotated, detections, infer_ms):
        # Resize output before JPEG encoding for speed
        if OUTPUT_MAX_WIDTH > 0 and annotated.shape[1] > OUTPUT_MAX_WIDTH:
            scale = OUTPUT_MAX_WIDTH / annotated.shape[1]
            new_h = int(annotated.shape[0] * scale)
            annotated = cv2.resize(annotated, (OUTPUT_MAX_WIDTH, new_h),
                                   interpolation=cv2.INTER_NEAREST)
        _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        with self._output_lock:
            self._annotated_frame = annotated
            self._jpeg_buffer = buf.tobytes()
            self._detections = detections
            self._infer_ms = infer_ms

    def get_last_detections(self):
        """Return last detections for reuse on skipped frames."""
        with self._output_lock:
            return self._detections.copy(), self._infer_ms

    def get_jpeg(self):
        with self._output_lock:
            return self._jpeg_buffer

    def get_detections_info(self):
        with self._output_lock:
            return {
                "detections": self._detections,
                "fps_capture": round(self._fps_capture, 1),
                "fps_infer": round(self._fps_infer, 1),
                "infer_ms": round(self._infer_ms, 1),
                "mode": self.mode,
                "source": self.source,
            }

state = DetectionState()


def parse_nms_output(raw_output, img_w, img_h):
    detections = []
    for cls_id, cls_dets in enumerate(raw_output):
        arr = np.array(cls_dets)
        if arr.size == 0:
            continue
        for det in arr:
            score = det[4]
            if score < CONF_THRESHOLD:
                continue
            y_min, x_min, y_max, x_max = det[0], det[1], det[2], det[3]
            x1 = int(x_min * img_w)
            y1 = int(y_min * img_h)
            x2 = int(x_max * img_w)
            y2 = int(y_max * img_h)
            detections.append({
                "class_id": cls_id,
                "class_name": COCO_CLASSES[cls_id],
                "confidence": round(float(score), 3),
                "bbox": [x1, y1, x2, y2]
            })
    return detections


def draw_detections(frame, detections):
    for d in detections:
        cls_id = d["class_id"]
        color = COLORS[cls_id]
        x1, y1, x2, y2 = d["bbox"]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def capture_loop():
    """Thread 1: Capture frames as fast as possible, store latest only."""
    while True:
        src = state.source
        if src == "camera":
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"Cannot open source: {src}")
            time.sleep(1)
            continue

        # Warm up camera
        if src == "camera":
            for _ in range(5):
                cap.read()

        state.running = True
        state.restart_flag = False
        frame_count = 0
        t_start = time.time()

        while not state.restart_flag:
            ret, frame = cap.read()
            if not ret:
                if src != "camera":
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            state.put_raw_frame(frame)
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed > 0:
                state._fps_capture = frame_count / elapsed
            # Reset counter every 5s to get recent average
            if elapsed > 5:
                frame_count = 0
                t_start = time.time()

        cap.release()
        state.running = False


def inference_loop():
    """Thread 2: Run Hailo YOLO on latest frame, skip stale frames."""
    hef = HEF(HEF_PATH)
    params = VDevice.create_params()

    with VDevice(params) as vdevice:
        cp = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
        ngs = vdevice.configure(hef, cp)
        ng = ngs[0]
        ivp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        ovp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        input_info = hef.get_input_vstream_infos()[0]

        with ng.activate():
            with InferVStreams(ng, ivp, ovp) as pipeline:
                last_seq = -1
                frame_count = 0
                skip_counter = 0  # counts frames since last inference
                t_start = time.time()

                while True:
                    frame, seq = state.get_raw_frame()

                    if frame is None or seq == last_seq:
                        time.sleep(0.005)
                        continue

                    last_seq = seq

                    if state.mode == "passthrough":
                        # No YOLO - just pass through raw frame with FPS overlay
                        annotated = frame.copy()
                        cv2.putText(annotated,
                                    f"PASSTHROUGH | Capture: {state._fps_capture:.1f} FPS",
                                    (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        state.put_output(annotated, [], 0)
                        continue

                    h, w = frame.shape[:2]
                    skip_counter += 1

                    if skip_counter >= INFER_EVERY_N:
                        # --- Full inference frame ---
                        skip_counter = 0
                        resized = cv2.resize(frame, (640, 640),
                                             interpolation=cv2.INTER_NEAREST)
                        input_data = np.expand_dims(resized, axis=0)

                        t_infer = time.time()
                        results = pipeline.infer({input_info.name: input_data})
                        infer_ms = (time.time() - t_infer) * 1000

                        output_name = list(results.keys())[0]
                        raw_output = results[output_name][0]
                        detections = parse_nms_output(raw_output, w, h)
                    else:
                        # --- Skip frame: reuse previous detections ---
                        detections, infer_ms = state.get_last_detections()

                    annotated = draw_detections(frame.copy(), detections)

                    # FPS tracking
                    frame_count += 1
                    elapsed = time.time() - t_start
                    if elapsed > 0:
                        state._fps_infer = frame_count / elapsed
                    if elapsed > 5:
                        frame_count = 0
                        t_start = time.time()

                    # Overlay diagnostics
                    skip_info = f"Skip:{INFER_EVERY_N-1}/{INFER_EVERY_N}"
                    cv2.putText(annotated,
                                f"Cap:{state._fps_capture:.0f} Inf:{state._fps_infer:.0f}FPS {infer_ms:.0f}ms {skip_info}",
                                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    state.put_output(annotated, detections, infer_ms)


def generate_mjpeg():
    """MJPEG stream generator - serves pre-encoded JPEG frames."""
    while True:
        jpeg = state.get_jpeg()
        if jpeg is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg + b'\r\n')
        time.sleep(0.03)


# --- Routes ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detections')
def get_detections():
    return jsonify(state.get_detections_info())


@app.route('/sources')
def get_sources():
    videos = sorted(glob.glob(os.path.join(VIDEO_DIR, "*.mp4")))
    videos += sorted(glob.glob(os.path.join(VIDEO_DIR, "*.avi")))
    sources = [{"id": "camera", "name": "USB Camera (Live)"}]
    for v in videos:
        sources.append({"id": v, "name": os.path.basename(v)})
    return jsonify(sources)


@app.route('/set_source', methods=['POST'])
def set_source():
    src = request.json.get('source', 'camera')
    state.source = src
    state.restart_flag = True
    return jsonify({"ok": True, "source": src})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    mode = request.json.get('mode', 'yolo')
    if mode in ('yolo', 'passthrough'):
        state.mode = mode
    return jsonify({"ok": True, "mode": state.mode})


# --- HTML Template ---
HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hailo YOLOv8 Live Detection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f1923; color: #e0e0e0; }
  .header {
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    padding: 1rem 2rem;
    display: flex; align-items: center; justify-content: space-between;
  }
  .header h1 { font-size: 1.4rem; color: #fff; }
  .header .badge-group { display: flex; gap: 8px; align-items: center; }
  .badge {
    padding: 4px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;
  }
  .badge-capture { background: #00e676; color: #000; }
  .badge-infer { background: #ffc107; color: #000; }
  .badge-latency { background: #ff9800; color: #000; }
  .main { display: flex; gap: 1rem; padding: 1rem; height: calc(100vh - 60px); }
  .video-panel { flex: 3; display: flex; flex-direction: column; }
  .video-panel img {
    width: 640px; height: 480px; border-radius: 8px; background: #000;
    object-fit: contain;
  }
  .side-panel { flex: 1; display: flex; flex-direction: column; gap: 1rem; min-width: 280px; }
  .card {
    background: #1a2632; border-radius: 8px; padding: 1rem;
    border: 1px solid #2a3a4a;
  }
  .card h3 { color: #64b5f6; margin-bottom: 0.75rem; font-size: 0.95rem; }
  select, .mode-btn {
    width: 100%; padding: 8px 12px; border-radius: 6px;
    background: #0f1923; color: #e0e0e0; border: 1px solid #3a4a5a;
    font-size: 0.9rem; cursor: pointer; margin-bottom: 6px;
  }
  select:focus { outline: none; border-color: #1a73e8; }
  .mode-btn { text-align: center; font-weight: 600; transition: all 0.2s; }
  .mode-btn:hover { border-color: #1a73e8; }
  .mode-btn.active { background: #1a73e8; border-color: #1a73e8; color: #fff; }
  .det-list {
    max-height: calc(100vh - 340px); overflow-y: auto;
    font-size: 0.85rem;
  }
  .det-item {
    display: flex; justify-content: space-between; align-items: center;
    padding: 6px 8px; border-radius: 4px; margin-bottom: 4px;
    background: #0f1923;
  }
  .det-item .cls-name { font-weight: 600; }
  .det-item .conf {
    background: #1a73e8; padding: 2px 8px; border-radius: 10px;
    font-size: 0.8rem; font-weight: 600;
  }
  .det-item .conf.high { background: #00c853; color: #000; }
  .det-item .conf.mid  { background: #ffc107; color: #000; }
  .det-item .conf.low  { background: #ff5722; }
  .stats { display: flex; gap: 0.5rem; flex-wrap: wrap; }
  .stat-chip {
    background: #0f1923; padding: 4px 10px; border-radius: 12px;
    font-size: 0.8rem; color: #90caf9;
  }
  .diag-bar {
    margin-top: 8px; padding: 8px; background: #0f1923; border-radius: 6px;
    font-size: 0.8rem; line-height: 1.6;
  }
  .diag-bar .label { color: #90caf9; }
  .diag-bar .val { font-weight: 700; }
  .diag-bar .val.good { color: #00e676; }
  .diag-bar .val.warn { color: #ffc107; }
  .diag-bar .val.bad { color: #ff5722; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #1a2632; }
  ::-webkit-scrollbar-thumb { background: #3a4a5a; border-radius: 3px; }
</style>
</head>
<body>
  <div class="header">
    <h1>Hailo-8 YOLOv8s Live Detection</h1>
    <div class="badge-group">
      <span class="badge badge-capture" id="capBadge">Capture: --</span>
      <span class="badge badge-infer" id="infBadge">Infer: --</span>
      <span class="badge badge-latency" id="latBadge">-- ms</span>
    </div>
  </div>
  <div class="main">
    <div class="video-panel">
      <img id="videoStream" src="/video_feed" alt="Live Stream">
    </div>
    <div class="side-panel">
      <div class="card">
        <h3>Video Source</h3>
        <select id="sourceSelect" onchange="changeSource()">
          <option value="camera">Loading...</option>
        </select>
        <h3 style="margin-top:10px;">Mode (for diagnosis)</h3>
        <div style="display:flex; gap:6px;">
          <button class="mode-btn active" id="btnYolo" onclick="setMode('yolo')">YOLO</button>
          <button class="mode-btn" id="btnPass" onclick="setMode('passthrough')">Passthrough</button>
        </div>
        <div class="diag-bar" id="diagBar">
          <span class="label">Diagnosis:</span> Switch to <b>Passthrough</b> to see raw camera FPS without YOLO.<br>
          Compare capture FPS in both modes to find the bottleneck.
        </div>
      </div>
      <div class="card" style="flex:1;">
        <h3>Detections <span id="detCount" class="stat-chip">0</span></h3>
        <div class="stats" id="statsArea"></div>
        <div class="det-list" id="detList" style="margin-top:0.5rem;"></div>
      </div>
    </div>
  </div>
<script>
let currentMode = 'yolo';

async function loadSources() {
  const res = await fetch('/sources');
  const sources = await res.json();
  const sel = document.getElementById('sourceSelect');
  sel.innerHTML = '';
  sources.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = s.name;
    sel.appendChild(opt);
  });
}

async function changeSource() {
  const src = document.getElementById('sourceSelect').value;
  await fetch('/set_source', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: src})
  });
}

async function setMode(mode) {
  await fetch('/set_mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: mode})
  });
  currentMode = mode;
  document.getElementById('btnYolo').classList.toggle('active', mode === 'yolo');
  document.getElementById('btnPass').classList.toggle('active', mode === 'passthrough');
}

function confClass(c) {
  if (c >= 0.7) return 'high';
  if (c >= 0.5) return 'mid';
  return 'low';
}

function fpsClass(fps) {
  if (fps >= 15) return 'good';
  if (fps >= 8) return 'warn';
  return 'bad';
}

async function updateDetections() {
  try {
    const res = await fetch('/detections');
    const data = await res.json();

    // Update badges
    document.getElementById('capBadge').textContent = 'Capture: ' + data.fps_capture + ' FPS';
    document.getElementById('infBadge').textContent = 'Infer: ' + data.fps_infer + ' FPS';
    document.getElementById('latBadge').textContent = data.infer_ms + ' ms';
    document.getElementById('detCount').textContent = data.detections.length;

    // Diagnosis
    const diag = document.getElementById('diagBar');
    if (data.mode === 'passthrough') {
      diag.innerHTML = '<span class="label">Passthrough mode</span> - YOLO off<br>' +
        'Camera raw FPS: <span class="val ' + fpsClass(data.fps_capture) + '">' + data.fps_capture + '</span><br>' +
        'If this is also low, the bottleneck is <b>camera/USB</b> or <b>Flask streaming</b>.';
    } else {
      let bottleneck = '';
      if (data.fps_capture > data.fps_infer * 2) {
        bottleneck = 'Bottleneck: <span class="val bad">YOLO inference</span> (camera is fast, inference is slow)';
      } else if (data.fps_capture < 15) {
        bottleneck = 'Bottleneck: <span class="val warn">Camera capture</span> (camera itself is slow)';
      } else {
        bottleneck = 'Pipeline is <span class="val good">balanced</span>';
      }
      diag.innerHTML = '<span class="label">YOLO mode</span><br>' +
        'Capture: <span class="val ' + fpsClass(data.fps_capture) + '">' + data.fps_capture + '</span> | ' +
        'Inference: <span class="val ' + fpsClass(data.fps_infer) + '">' + data.fps_infer + '</span><br>' +
        bottleneck;
    }

    // Class summary
    const counts = {};
    data.detections.forEach(d => {
      counts[d.class_name] = (counts[d.class_name] || 0) + 1;
    });
    const statsHtml = Object.entries(counts)
      .map(([k,v]) => `<span class="stat-chip">${k}: ${v}</span>`).join('');
    document.getElementById('statsArea').innerHTML = statsHtml;

    // Detection list
    const listHtml = data.detections
      .sort((a,b) => b.confidence - a.confidence)
      .map(d => `<div class="det-item">
        <span class="cls-name">${d.class_name}</span>
        <span class="conf ${confClass(d.confidence)}">${(d.confidence*100).toFixed(1)}%</span>
      </div>`).join('');
    document.getElementById('detList').innerHTML = listHtml || '<div style="color:#666;padding:8px;">No detections</div>';
  } catch(e) {}
}

loadSources();
setInterval(updateDetections, 500);
</script>
</body>
</html>'''


if __name__ == '__main__':
    # Start capture thread
    t_cap = threading.Thread(target=capture_loop, daemon=True)
    t_cap.start()
    # Start inference thread
    t_inf = threading.Thread(target=inference_loop, daemon=True)
    t_inf.start()
    print("Starting Flask server on http://0.0.0.0:5000")
    print("Use Passthrough mode to diagnose if bottleneck is YOLO or camera/streaming")
    app.run(host='0.0.0.0', port=5000, threaded=True)
