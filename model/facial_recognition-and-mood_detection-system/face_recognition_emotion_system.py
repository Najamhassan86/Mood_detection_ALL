import cv2
import numpy as np
import pickle
import time
import base64
import random
import requests
import os  # ✅ NEW
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

# =========================
# UniFace
# =========================
from uniface import RetinaFace, ArcFace, Emotion, compute_similarity
from uniface.constants import RetinaFaceWeights

# =========================
# ONNX Inference (optional)
# =========================
try:
    from onnx_inference import ONNXArcFace, ONNXEmotion, load_hybrid_models
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("⚠️  ONNX models not available. Install: pip install onnx onnxruntime")

# =========================
# BACKEND CONFIG
# =========================
# ✅ CHANGED: local backend
BACKEND_BASE = "http://127.0.0.1:8000"
FRAME_UPLOAD_ENDPOINT = f"{BACKEND_BASE}/api/stream/frame"
BATCH_ENDPOINT = f"{BACKEND_BASE}/api/emotions/batch"

DEVICE_ID = "jetson_1"

FRAME_SEND_INTERVAL = 0.5   # seconds
BATCH_SEND_INTERVAL = 5.0   # seconds

# ✅ NEW: terminal table refresh rate
TABLE_REFRESH_INTERVAL = 1.0  # seconds


# =========================
# UTILS
# =========================
def draw_emotion_label(image, bbox, emotion, confidence):
    x1, y1 = int(bbox[0]), int(bbox[1])
    text = f"{emotion} ({confidence:.2f})"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

    cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255, 0, 0), -1)
    cv2.putText(image, text, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


def send_frame_to_backend(frame, fps=0.0):
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    if not ok:
        print("[WARN] Failed to encode frame")
        return

    frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    payload = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "frame_b64": frame_b64,
        "fps": fps,  # ✅ Include FPS in payload
    }

    try:
        r = requests.post(FRAME_UPLOAD_ENDPOINT, json=payload, timeout=10)
        print(f"[DEBUG] Frame upload HTTP {r.status_code} | FPS: {fps:.1f}")
        if r.ok:
            size_kb = len(frame_b64) // 1024
            print(f"[INFO] Frame uploaded successfully ({size_kb} KB)")
        else:
            print(f"[ERROR] Upload failed with status {r.status_code}: {r.text[:1000]}")
            try:
                r.raise_for_status()
            except Exception:
                pass
    except Exception as e:
        print("[WARN] Frame upload failed:", e)


def send_dummy_batch():
    payload = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "people": [
            {
                "person_id": "1",
                "cumulative": {
                    "happy": random.uniform(10, 200),
                    "sad": random.uniform(0, 100),
                    "angry": random.uniform(0, 50),
                },
            }
        ],
    }

    try:
        requests.post(BATCH_ENDPOINT, json=payload, timeout=5)
    except Exception as e:
        print("[WARN] Batch upload failed:", e)


def send_stats_batch(stats: Dict[str, Dict], known_emotions: set):
    # We'll send cumulative totals for ALL registered people (scalable + consistent)
    payload_people = []

    # stable keys (backend supports these)
    keys = ["happy", "sad", "angry", "neutral", "unknown"]

    for person_name, st in stats.items():
        cumulative = {}
        for k in keys:
            cumulative[k] = float(st["moods"].get(k, 0.0))

        payload_people.append({
            "person_id": person_name,     # IMPORTANT: use registered name/ID
            "cumulative": cumulative
        })

    payload = {
        "device_id": DEVICE_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "people": payload_people,
    }

    try:
        r = requests.post(BATCH_ENDPOINT, json=payload, timeout=8)
        if not r.ok:
            print(f"[ERROR] Stats batch failed {r.status_code}: {r.text[:300]}")
    except Exception as e:
        print("[WARN] Stats batch upload failed:", e)


# =========================
# MAIN SYSTEM
# =========================
class FaceRecognitionEmotionSystem:

    def __init__(self,
                 database_path="database",
                 embeddings_path="embeddings.pkl",
                 similarity_threshold=0.4,
                 use_onnx=False):

        self.database_path = Path(database_path)
        self.embeddings_path = Path(embeddings_path)
        self.similarity_threshold = similarity_threshold
        self.use_onnx = use_onnx

        print("🔄 Loading models...")

        # Check if ONNX models should be used
        if use_onnx and ONNX_AVAILABLE:
            # Check if at least ArcFace ONNX model exists
            # (Emotion can fall back to TorchScript or UniFace)
            arcface_exists = Path("onnx_models/arcface_model.onnx").exists()

            if arcface_exists:
                print("🚀 Loading optimized models...")
                self.detector, self.recognizer, self.emotion_predictor = load_hybrid_models(use_onnx=True)
                print("✅ Optimized models loaded successfully")
            else:
                print("⚠️  ONNX models not found. Run 'python extract_all_models.py' first.")
                print("🔄 Falling back to UniFace PyTorch models...")
                self._load_uniface_models()
        else:
            # Use standard UniFace PyTorch models
            if use_onnx and not ONNX_AVAILABLE:
                print("⚠️  ONNX not available. Install: pip install onnx onnxruntime")
            print("🔄 Loading UniFace PyTorch models...")
            self._load_uniface_models()

        self.face_database: Dict[str, np.ndarray] = {}

        # ✅ NEW: stats store (created after load_database or registration)
        # shape:
        # self.stats[name] = {
        #   "total_time": float,
        #   "moods": {emotion: float},
        #   "current_emotion": str,
        #   "last_seen": float (epoch seconds) | None
        # }
        self.stats: Dict[str, Dict] = {}
        self._known_emotions = set()  # learned dynamically

        print("✅ Models loaded\n")

    def _load_uniface_models(self):
        """Load standard UniFace PyTorch models"""
        self.detector = RetinaFace(
            model_name=RetinaFaceWeights.MNET_V2,
            conf_thresh=0.6,
            input_size=(640, 640),
        )
        self.recognizer = ArcFace()
        if Emotion is None:
            raise RuntimeError(
                "UniFace 'Emotion' component is unavailable. Install PyTorch (e.g. `pip install torch torchvision`) and restart."
            )
        self.emotion_predictor = Emotion()

    # =========================
    # REGISTRATION
    # =========================
    def register_new_person(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("❌ Webcam not accessible")
            return

        poses = ["STRAIGHT", "LEFT", "RIGHT", "UP", "DOWN"]
        pose_index = 0
        captured = []
        last_capture = 0

        print("\n📸 Registration started")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detector.detect(frame)

            if len(faces) != 1:
                cv2.putText(frame, "Ensure ONLY ONE face",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("Registration", frame)
                cv2.waitKey(1)
                continue

            face = faces[0]
            landmarks = face["landmarks"]

            left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
            dx = nose[0] - (left_eye[0] + right_eye[0]) / 2
            dy = nose[1] - (left_eye[1] + right_eye[1]) / 2

            current_pose = poses[pose_index]
            pose_ok = (
                (current_pose == "STRAIGHT" and abs(dx) < 10 and abs(dy) < 10) or
                (current_pose == "LEFT" and dx > 15) or
                (current_pose == "RIGHT" and dx < -15) or
                (current_pose == "UP" and dy < -15) or
                (current_pose == "DOWN" and dy > 15)
            )

            if pose_ok and time.time() - last_capture > 0.8:
                captured.append(frame.copy())
                last_capture = time.time()
                print(f"✅ Captured: {current_pose}")
                pose_index += 1

            cv2.putText(frame, f"POSE: {current_pose}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.imshow("Registration", frame)

            if pose_index >= len(poses):
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(captured) < len(poses):
            print("❌ Registration incomplete")
            return

        name = input("Enter person's name/ID: ").strip()
        person_dir = self.database_path / name
        person_dir.mkdir(parents=True, exist_ok=True)

        embeddings = []
        for img in captured:
            faces = self.detector.detect(img)
            if faces:
                emb = self.recognizer.get_normalized_embedding(img, faces[0]["landmarks"])
                embeddings.append(emb)

        if embeddings:
            self.face_database[name] = np.mean(embeddings, axis=0)
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(self.face_database, f)

            # ✅ NEW: ensure stats row exists immediately for this person
            self._ensure_stats_for_person(name)

            print(f"✅ {name} registered successfully")

    # =========================
    # DATABASE
    # =========================
    def load_database(self):
        if not self.embeddings_path.exists():
            print("❌ No database found")
            return False
        with open(self.embeddings_path, "rb") as f:
            self.face_database = pickle.load(f)
        print(f"✅ Loaded {len(self.face_database)} people")

        # ✅ NEW: ensure stats for all registered people (even if not seen yet)
        for name in self.face_database.keys():
            self._ensure_stats_for_person(name)

        return True

    def recognize_face(self, embedding):
        best_name, best_score = "Unknown", 0.0
        for name, db_emb in self.face_database.items():
            sim = float(compute_similarity(embedding, db_emb).flatten()[0])
            if sim > best_score:
                best_name, best_score = name, sim
        return (best_name, best_score) if best_score >= self.similarity_threshold else ("Unknown", best_score)

    # =========================
    # ✅ NEW: STATS HELPERS
    # =========================
    def _ensure_stats_for_person(self, name: str):
        if name not in self.stats:
            self.stats[name] = {
                "total_time": 0.0,
                "moods": {},  # filled dynamically with seen emotions
                "current_emotion": "unknown",
                "last_seen": None,
            }

    def _add_time(self, name: str, emotion: str, dt: float, now_ts: float):
        """Accumulate dt seconds to total + emotion bucket for this person."""
        self._ensure_stats_for_person(name)

        # track known emotions dynamically
        emotion_key = (emotion or "unknown").lower()
        self._known_emotions.add(emotion_key)

        self.stats[name]["total_time"] += dt
        self.stats[name]["moods"][emotion_key] = self.stats[name]["moods"].get(emotion_key, 0.0) + dt
        self.stats[name]["current_emotion"] = emotion_key
        self.stats[name]["last_seen"] = now_ts

    def _render_terminal_table(self):
        """Print a clean table of all registered people and their time stats."""
        # clear terminal (Windows)
        os.system("cls" if os.name == "nt" else "clear")

        # stable emotion ordering: show these first if present
        preferred = ["happy", "sad", "angry", "neutral", "unknown"]
        emotions = [e for e in preferred if e in self._known_emotions] + \
                   sorted([e for e in self._known_emotions if e not in preferred])

        header = ["person", "on_cam(s)"] + [f"{e}(s)" for e in emotions] + ["current", "last_seen(s_ago)"]
        col_w = [18, 10] + [10] * len(emotions) + [10, 15]

        def fmt_row(values):
            return "  ".join(str(v).ljust(w)[:w] for v, w in zip(values, col_w))

        print("📊 LIVE PERSON EMOTION TIME STATS (registered people)\n")
        print(fmt_row(header))
        print("-" * (sum(col_w) + 2 * (len(col_w) - 1)))

        now_ts = time.time()
        for name in sorted(self.face_database.keys()):
            self._ensure_stats_for_person(name)
            st = self.stats[name]

            total = st["total_time"]
            mood_vals = []
            for e in emotions:
                mood_vals.append(st["moods"].get(e, 0.0))

            last_seen = st["last_seen"]
            last_seen_ago = f"{(now_ts - last_seen):.1f}" if last_seen is not None else "-"

            row = [
                name,
                f"{total:.1f}",
                *[f"{v:.1f}" for v in mood_vals],
                st["current_emotion"],
                last_seen_ago,
            ]
            print(fmt_row(row))

        print("\n(Press 'q' in the video window to stop)")

    # =========================
    # LIVE MODE
    # =========================
    def run_live(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("🎥 Live Recognition + Emotion + Streaming")

        last_frame_sent = 0
        last_batch_sent = 0

        # ✅ NEW: timing + table refresh bookkeeping
        prev_frame_time = time.time()
        last_table_print = 0

        # ✅ FPS calculation variables
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        fps_update_interval = 1.0  # Update FPS every second

        # ✅ ensure stats exist for all registered people
        for name in self.face_database.keys():
            self._ensure_stats_for_person(name)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            faces = self.detector.detect(frame)

            now = time.time()
            frame_dt = max(0.0, now - prev_frame_time)
            prev_frame_time = now

            # ✅ FPS calculation
            fps_frame_count += 1
            fps_elapsed = now - fps_start_time
            if fps_elapsed >= fps_update_interval:
                current_fps = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_start_time = now

            # ✅ NEW: collect who is present in THIS frame (avoid double-counting)
            present_people = {}  # name -> emotion

            for face in faces:
                landmarks = face["landmarks"]
                emb = self.recognizer.get_normalized_embedding(frame, landmarks)
                name, score = self.recognize_face(emb)
                emotion, emo_conf = self.emotion_predictor.predict(frame, landmarks)

                x1, y1, x2, y2 = map(int, face["bbox"])
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} ({score:.2f})",
                            (x1, y1 - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                draw_emotion_label(frame, face["bbox"], emotion, emo_conf)

                # ✅ NEW: stats only for recognized registered people
                if name != "Unknown":
                    present_people[name] = emotion

            # ✅ NEW: update stats ONCE per person per frame
            for name, emo in present_people.items():
                self._add_time(name=name, emotion=emo, dt=frame_dt, now_ts=now)

            # ✅ Display FPS on camera window (top-left corner)
            fps_text = f"Model FPS: {current_fps:.1f}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # existing send frame logic (unchanged)
            if now - last_frame_sent > FRAME_SEND_INTERVAL:
                send_frame_to_backend(frame, current_fps)
                last_frame_sent = now

            # send real stats batch
            if now - last_batch_sent > BATCH_SEND_INTERVAL:
                send_stats_batch(self.stats, self._known_emotions)
                last_batch_sent = now

            # ✅ NEW: print table every N seconds
            if now - last_table_print > TABLE_REFRESH_INTERVAL:
                self._render_terminal_table()
                last_table_print = now

            cv2.imshow("UniFace Live System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# =========================
# MENU
# =========================
def main():
    # Ask user to select model type
    print("\n" + "=" * 60)
    print("🤖 SELECT MODEL TYPE")
    print("=" * 60)
    print("1. PyTorch Models (UniFace - default)")
    print("2. ONNX Models (Faster inference - requires conversion)")
    print("=" * 60)

    model_choice = input("Select model type (1/2, default=1): ").strip()
    use_onnx = (model_choice == "2")

    if use_onnx:
        print("\n📌 ONNX Mode Selected")
        print("   Make sure you've extracted models first:")
        print("   1. python extract_all_models.py       (for ArcFace & RetinaFace)")
        print("   2. python save_emotion_torchscript.py (for Emotion)")
        print()
    else:
        print("\n📌 PyTorch Mode Selected (default)\n")

    system = FaceRecognitionEmotionSystem(use_onnx=use_onnx)

    while True:
        print("\n" + "=" * 60)
        print("📋 MENU")
        print("1. Register New Person")
        print("2. Load Existing Database")
        print("3. Start Live Recognition + Emotion + Streaming")
        print("4. Exit")
        print("=" * 60)

        choice = input("Select option: ").strip()

        if choice == "1":
            system.register_new_person()
        elif choice == "2":
            system.load_database()
        elif choice == "3":
            # ✅ Small note: make sure DB is loaded so stats includes all registered people
            if not system.face_database:
                print("⚠️ No database loaded. Load option (2) first so stats table includes all registered people.")
            system.run_live()
        elif choice == "4":
            print("👋 Bye")
            break
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
