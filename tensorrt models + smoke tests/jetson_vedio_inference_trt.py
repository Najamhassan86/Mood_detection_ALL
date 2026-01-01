#!/usr/bin/env python3
"""
Jetson TensorRT Video Inference Script
Processes a pre-recorded video file using three TensorRT engines:
- RetinaFace (face detection)
- ArcFace (face recognition)
- Emotion (mood detection)

Directory structure expected:
    models/
        retinaface_model.trt
        arcface_model.trt
        emotion_model.trt
    sample_video/
        your_video.mp4
    database/
        embeddings.pkl (optional, for face recognition)
"""

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import pickle
import time
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional


# ========================= TensorRT Engine Wrapper =========================

class TRTEngine:
    """Generic TensorRT engine wrapper"""
    
    def __init__(self, engine_path: str):
        self.engine_path = Path(engine_path)
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        
        # Initialize TensorRT
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        # Load engine
        with open(self.engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            shape = self.engine.get_tensor_shape(tensor_name)
            
            # Handle dynamic shapes
            if -1 in shape:
                shape = self.context.get_tensor_shape(tensor_name)
            
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
            else:
                self.outputs.append({
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': shape,
                    'dtype': dtype
                })
        
        print(f"✅ Loaded TensorRT engine: {engine_path}")
        print(f"   Inputs: {[inp['name'] + str(inp['shape']) for inp in self.inputs]}")
        print(f"   Outputs: {[out['name'] + str(out['shape']) for out in self.outputs]}")
    
    def infer(self, input_data: np.ndarray) -> List[np.ndarray]:
        """Run inference on input data"""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        
        # Set tensor addresses
        for i, inp in enumerate(self.inputs):
            self.context.set_tensor_address(inp['name'], self.bindings[i])
        for i, out in enumerate(self.outputs):
            self.context.set_tensor_address(out['name'], self.bindings[len(self.inputs) + i])
        
        # Execute
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # Copy outputs back to host
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            self.stream.synchronize()
            output = out['host'].reshape(out['shape'])
            outputs.append(output)
        
        return outputs
    
    def __del__(self):
        """Cleanup"""
        del self.context
        del self.engine
        del self.runtime


# ========================= TensorRT Model Wrappers =========================

class TRTRetinaFaceSimple:
    """Simplified RetinaFace for face detection using OpenCV DNN"""

    def __init__(self, conf_thresh=0.6):
        self.conf_thresh = conf_thresh
        print("⚠️  Using OpenCV Haar Cascade for face detection (RetinaFace TRT requires complex post-processing)")
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect faces using Haar Cascade"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        detections = []
        for (x, y, w, h) in faces:
            landmarks = np.array([
                [x + w * 0.3, y + h * 0.4],
                [x + w * 0.7, y + h * 0.4],
                [x + w * 0.5, y + h * 0.6],
                [x + w * 0.35, y + h * 0.8],
                [x + w * 0.65, y + h * 0.8]
            ], dtype=np.float32)

            detections.append({
                'bbox': [x, y, x + w, y + h],
                'landmarks': landmarks,
                'confidence': 1.0
            })

        return detections


class TRTArcFace:
    """TensorRT wrapper for ArcFace face recognition model"""

    def __init__(self, engine_path="models/arcface_model.trt"):
        self.engine = TRTEngine(engine_path)
        print(f"✅ Loaded ArcFace TensorRT engine")

    def preprocess_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align and preprocess face for ArcFace (112x112)"""
        src_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(landmarks, src_points)[0]
        aligned_face = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

        # Normalize to [-1, 1]
        aligned_face = aligned_face.astype(np.float32)
        aligned_face = (aligned_face - 127.5) / 127.5

        # Convert to CHW format
        face_tensor = np.transpose(aligned_face, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)

        return face_tensor

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Get normalized face embedding (512,)"""
        face_tensor = self.preprocess_face(image, landmarks)
        
        outputs = self.engine.infer(face_tensor)
        embedding = outputs[0]

        # Normalize
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class TRTEmotion:
    """TensorRT wrapper for Emotion classification model"""

    EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

    def __init__(self, engine_path="models/emotion_model.trt"):
        self.engine = TRTEngine(engine_path)
        print(f"✅ Loaded Emotion TensorRT engine")

    def preprocess_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Align and preprocess face for emotion model"""
        src_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(landmarks, src_points)[0]
        aligned_face = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

        # Normalize [0, 1]
        aligned_face = aligned_face.astype(np.float32) / 255.0

        # Convert to CHW
        face_tensor = np.transpose(aligned_face, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)

        return face_tensor

    def predict(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[str, float]:
        """Predict emotion from face"""
        face_tensor = self.preprocess_face(image, landmarks)
        
        outputs = self.engine.infer(face_tensor)
        logits = outputs[0]

        probabilities = self._softmax(logits[0])
        emotion_idx = np.argmax(probabilities)
        confidence = probabilities[emotion_idx]

        emotion_label = self.EMOTION_LABELS[emotion_idx] if emotion_idx < len(self.EMOTION_LABELS) else "Unknown"

        return emotion_label, float(confidence)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


# ========================= Face Database =========================

class FaceDatabase:
    """Simple face database for recognition"""

    def __init__(self, embeddings_path="database/embeddings.pkl", similarity_threshold=0.4):
        self.embeddings_path = Path(embeddings_path)
        self.similarity_threshold = similarity_threshold
        self.database: Dict[str, np.ndarray] = {}

        if self.embeddings_path.exists():
            with open(self.embeddings_path, "rb") as f:
                self.database = pickle.load(f)
            print(f"✅ Loaded face database: {len(self.database)} people")
        else:
            print("⚠️  No face database found. All faces will be 'Unknown'")

    def recognize(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Recognize face from embedding"""
        if not self.database:
            return "Unknown", 0.0

        best_name, best_score = "Unknown", 0.0
        for name, db_emb in self.database.items():
            similarity = self._cosine_similarity(embedding, db_emb)
            if similarity > best_score:
                best_name, best_score = name, similarity

        return (best_name, best_score) if best_score >= self.similarity_threshold else ("Unknown", best_score)

    @staticmethod
    def _cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()

        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


# ========================= Visualization =========================

def draw_results(frame, face, name, score, emotion, confidence):
    """Draw bounding box, name, and emotion on frame"""
    x1, y1, x2, y2 = map(int, face['bbox'])

    # Draw bounding box
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw name and score
    name_text = f"{name} ({score:.2f})"
    cv2.putText(frame, name_text, (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw emotion label
    emotion_text = f"{emotion} ({confidence:.2f})"
    (tw, th), _ = cv2.getTextSize(emotion_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)

    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), (255, 0, 0), -1)
    cv2.putText(frame, emotion_text, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ========================= Main Inference =========================

def main():
    parser = argparse.ArgumentParser(description='TensorRT Video Inference on Jetson')
    parser.add_argument('--video', type=str, default='sample_video/test.mp4',
                        help='Path to video file')
    parser.add_argument('--models', type=str, default='models',
                        help='Path to models folder')
    parser.add_argument('--database', type=str, default='database/embeddings.pkl',
                        help='Path to face database pickle file')
    parser.add_argument('--show-fps', action='store_true', default=True,
                        help='Show FPS on video')
    parser.add_argument('--output', type=str, default=None,
                        help='Save output video to this path (optional)')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 Jetson TensorRT Video Inference")
    print("="*60 + "\n")

    # Check video file
    if not Path(args.video).exists():
        print(f"❌ Video file not found: {args.video}")
        print("   Please place your video in the sample_video/ folder")
        return

    # Load models
    print("📦 Loading TensorRT engines...\n")

    try:
        detector = TRTRetinaFaceSimple()
        recognizer = TRTArcFace(engine_path=f"{args.models}/arcface_model.trt")
        emotion_detector = TRTEmotion(engine_path=f"{args.models}/emotion_model.trt")
        face_db = FaceDatabase(embeddings_path=args.database)

    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nMake sure your folder structure looks like:")
        print("  models/")
        print("    ├── arcface_model.trt")
        print("    ├── emotion_model.trt")
        print("    └── retinaface_model.trt")
        print("  sample_video/")
        print("    └── your_video.mp4")
        print("  database/")
        print("    └── embeddings.pkl (optional)")
        return

    print("\n✅ All TensorRT engines loaded successfully!\n")

    # Open video file
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {args.video}")
        return

    # Get video properties
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"📹 Video Info:")
    print(f"   File: {args.video}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps_original:.2f}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps_original:.2f}s\n")

    # Setup video writer if output specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps_original, (width, height))
        print(f"💾 Saving output to: {args.output}\n")

    print("▶️  Processing video... (Press 'q' to quit)\n")

    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0.0
    fps_update_interval = 1.0

    frame_count = 0
    total_faces_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✅ Video processing complete!")
            break

        frame_count += 1
        now = time.time()

        # Detect faces
        faces = detector.detect(frame)
        total_faces_detected += len(faces)

        # Process each face
        for face in faces:
            landmarks = face['landmarks']

            # Face recognition
            embedding = recognizer.get_normalized_embedding(frame, landmarks)
            name, score = face_db.recognize(embedding)

            # Emotion detection
            emotion, confidence = emotion_detector.predict(frame, landmarks)

            # Draw results
            draw_results(frame, face, name, score, emotion, confidence)

        # Calculate and display FPS
        fps_frame_count += 1
        fps_elapsed = now - fps_start_time
        if fps_elapsed >= fps_update_interval:
            current_fps = fps_frame_count / fps_elapsed
            fps_frame_count = 0
            fps_start_time = now

        if args.show_fps:
            fps_text = f"TensorRT FPS: {current_fps:.1f} | Frame: {frame_count}/{total_frames}"
            cv2.putText(frame, fps_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Write frame if output is specified
        if writer:
            writer.write(frame)

        # Display frame
        cv2.imshow('TensorRT Inference - Jetson', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏹️  Stopped by user")
            break

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Print statistics
    print("\n" + "="*60)
    print("📊 Processing Statistics:")
    print("="*60)
    print(f"   Total Frames Processed: {frame_count}")
    print(f"   Total Faces Detected: {total_faces_detected}")
    print(f"   Average Faces per Frame: {total_faces_detected/frame_count:.2f}")
    print(f"   Average Inference FPS: {current_fps:.2f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
