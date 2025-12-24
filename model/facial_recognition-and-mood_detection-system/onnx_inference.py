"""
ONNX Inference Module
Provides wrapper classes for running inference with ONNX models
Also supports TorchScript models for Emotion (already optimized!)
"""

import cv2
import numpy as np
import onnxruntime as ort
import torch
from pathlib import Path
from typing import Tuple, List, Dict, Optional


class ONNXArcFace:
    """ONNX wrapper for ArcFace face recognition model"""

    def __init__(self, model_path="onnx_models/arcface_model.onnx"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']  # Use CPU, change to CUDAExecutionProvider for GPU
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✅ Loaded ArcFace ONNX model: {model_path}")

    def preprocess_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ArcFace model

        Args:
            image: Original image
            landmarks: 5 facial landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)

        Returns:
            Preprocessed face tensor (1, 3, 112, 112)
        """
        # Standard face alignment for ArcFace (112x112)
        src_points = np.array([
            [38.2946, 51.6963],  # left eye
            [73.5318, 51.5014],  # right eye
            [56.0252, 71.7366],  # nose
            [41.5493, 92.3655],  # left mouth
            [70.7299, 92.2041]   # right mouth
        ], dtype=np.float32)

        # Get transformation matrix
        tform = cv2.estimateAffinePartial2D(landmarks, src_points)[0]

        # Warp face to 112x112
        aligned_face = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

        # Normalize to [-1, 1]
        aligned_face = aligned_face.astype(np.float32)
        aligned_face = (aligned_face - 127.5) / 127.5

        # Convert to CHW format and add batch dimension
        face_tensor = np.transpose(aligned_face, (2, 0, 1))  # HWC -> CHW
        face_tensor = np.expand_dims(face_tensor, axis=0)  # Add batch dimension

        return face_tensor

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Get normalized face embedding

        Args:
            image: Original image
            landmarks: 5 facial landmarks

        Returns:
            Normalized embedding vector (512,)
        """
        # Preprocess face
        face_tensor = self.preprocess_face(image, landmarks)

        # Run inference
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: face_tensor}
        )[0]

        # Normalize embedding
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


class ONNXEmotion:
    """ONNX wrapper for Emotion classification model"""

    # UniFace emotion labels (EXACT order from UniFace library)
    EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

    def __init__(self, model_path="onnx_models/emotion_model.onnx"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(f"✅ Loaded Emotion ONNX model: {model_path}")

    def preprocess_face(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Preprocess face for emotion model

        Args:
            image: Original image
            landmarks: 5 facial landmarks

        Returns:
            Preprocessed face tensor (1, 3, 112, 112)
        """
        # Align face (similar to ArcFace)
        src_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(landmarks, src_points)[0]
        aligned_face = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

        # Normalize
        aligned_face = aligned_face.astype(np.float32) / 255.0

        # Convert to CHW format and add batch dimension
        face_tensor = np.transpose(aligned_face, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)

        return face_tensor

    def predict(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict emotion from face

        Args:
            image: Original image
            landmarks: 5 facial landmarks

        Returns:
            (emotion_label, confidence)
        """
        # Preprocess face
        face_tensor = self.preprocess_face(image, landmarks)

        # Run inference
        logits = self.session.run(
            [self.output_name],
            {self.input_name: face_tensor}
        )[0]

        # Get prediction
        probabilities = self._softmax(logits[0])
        emotion_idx = np.argmax(probabilities)
        confidence = probabilities[emotion_idx]

        emotion_label = self.EMOTION_LABELS[emotion_idx] if emotion_idx < len(self.EMOTION_LABELS) else "unknown"

        return emotion_label, float(confidence)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class TorchScriptEmotion:
    """TorchScript wrapper for Emotion classification model (already optimized!)"""

    # UniFace emotion labels (EXACT order from UniFace library)
    EMOTION_LABELS = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']

    def __init__(self, model_path="onnx_models/emotion_model.pt"):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")

        # Load TorchScript model
        self.model = torch.jit.load(str(self.model_path))
        self.model.eval()

        print(f"✅ Loaded Emotion TorchScript model: {model_path}")
        print(f"   Note: TorchScript is already optimized (~95% as fast as ONNX)!")

    def preprocess_face(self, image: np.ndarray, landmarks: np.ndarray) -> torch.Tensor:
        """
        Preprocess face for emotion model

        Args:
            image: Original image
            landmarks: 5 facial landmarks

        Returns:
            Preprocessed face tensor (1, 3, 112, 112)
        """
        # Align face
        src_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)

        tform = cv2.estimateAffinePartial2D(landmarks, src_points)[0]
        aligned_face = cv2.warpAffine(image, tform, (112, 112), borderValue=0.0)

        # Normalize
        aligned_face = aligned_face.astype(np.float32) / 255.0

        # Convert to CHW format and add batch dimension
        face_tensor = np.transpose(aligned_face, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)

        # Convert to PyTorch tensor
        face_tensor = torch.from_numpy(face_tensor).float()

        return face_tensor

    def predict(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Predict emotion from face

        Args:
            image: Original image
            landmarks: 5 facial landmarks

        Returns:
            (emotion_label, confidence)
        """
        # Preprocess face
        face_tensor = self.preprocess_face(image, landmarks)

        # Run inference
        with torch.no_grad():
            logits = self.model(face_tensor)

        # Convert to numpy for post-processing
        logits = logits.numpy()[0]

        # Get prediction
        probabilities = self._softmax(logits)
        emotion_idx = np.argmax(probabilities)
        confidence = probabilities[emotion_idx]

        emotion_label = self.EMOTION_LABELS[emotion_idx] if emotion_idx < len(self.EMOTION_LABELS) else "unknown"

        return emotion_label, float(confidence)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Apply softmax to logits"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()


class ONNXRetinaFace:
    """ONNX wrapper for RetinaFace detection model"""

    def __init__(self, model_path="onnx_models/retinaface_model.onnx",
                 conf_thresh=0.6, input_size=(640, 640)):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.conf_thresh = conf_thresh
        self.input_size = input_size

        # Load ONNX model
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=['CPUExecutionProvider']
        )

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name

        print(f"✅ Loaded RetinaFace ONNX model: {model_path}")
        print(f"⚠️  Note: RetinaFace ONNX inference requires custom post-processing")
        print(f"   Consider keeping UniFace RetinaFace or implementing full ONNX pipeline")

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image

        Note: This is a simplified placeholder. Full RetinaFace ONNX inference
        requires complex post-processing (anchor generation, NMS, etc.)

        For production, either:
        1. Keep using UniFace RetinaFace (PyTorch)
        2. Implement full ONNX post-processing pipeline

        Args:
            image: Input image

        Returns:
            List of face detections with bbox and landmarks
        """
        # Preprocess image
        img_h, img_w = image.shape[:2]
        input_tensor = self._preprocess(image)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Post-processing (simplified - needs full implementation)
        # RetinaFace outputs: [loc, conf, landmarks]
        # This requires anchor generation, NMS, coordinate transformation, etc.

        print("⚠️  RetinaFace ONNX post-processing not fully implemented")
        print("   Falling back to UniFace RetinaFace recommended")

        return []

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for RetinaFace"""
        # Resize to input size
        img_resized = cv2.resize(image, self.input_size)

        # Normalize
        img_tensor = img_resized.astype(np.float32)
        img_tensor -= np.array([104, 117, 123])  # RetinaFace mean subtraction

        # CHW format
        img_tensor = np.transpose(img_tensor, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)

        return img_tensor


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Similarity score (0 to 1)
    """
    dot_product = np.dot(emb1, emb2)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


# =========================
# Hybrid Mode: Use ONNX for ArcFace & Emotion, keep UniFace RetinaFace
# =========================
def load_hybrid_models(use_onnx=True):
    """
    Load models in hybrid mode:
    - RetinaFace: UniFace (PyTorch) - complex post-processing
    - ArcFace: ONNX - fast inference
    - Emotion: ONNX - fast inference

    Args:
        use_onnx: If True, use ONNX models for ArcFace and Emotion

    Returns:
        (detector, recognizer, emotion_predictor)
    """
    from uniface import RetinaFace
    from uniface.constants import RetinaFaceWeights

    # Always use UniFace RetinaFace (best detection quality)
    detector = RetinaFace(
        model_name=RetinaFaceWeights.MNET_V2,
        conf_thresh=0.6,
        input_size=(640, 640)
    )

    if use_onnx:
        # Use ONNX models for faster inference
        recognizer = ONNXArcFace()

        # Try loading Emotion model with fallback chain:
        # ONNX -> TorchScript -> UniFace
        try:
            emotion_predictor = ONNXEmotion()
            print("🚀 Using ONNX models for ArcFace and Emotion")
        except FileNotFoundError:
            print("⚠️  emotion_model.onnx not found, trying TorchScript...")
            try:
                emotion_predictor = TorchScriptEmotion()
                print("🚀 Using ONNX for ArcFace, TorchScript for Emotion")
            except FileNotFoundError:
                print("⚠️  emotion_model.pt not found, using UniFace Emotion")
                from uniface import Emotion
                emotion_predictor = Emotion()
                print("🚀 Using ONNX for ArcFace, UniFace PyTorch for Emotion")
    else:
        # Fallback to UniFace PyTorch models
        from uniface import ArcFace, Emotion
        recognizer = ArcFace()
        emotion_predictor = Emotion()
        print("🔄 Using UniFace PyTorch models")

    return detector, recognizer, emotion_predictor


if __name__ == "__main__":
    print("ONNX Inference Module")
    print("Usage: Import and use ONNXArcFace, ONNXEmotion classes")
    print("\nExample:")
    print("  from onnx_inference import load_hybrid_models")
    print("  detector, recognizer, emotion = load_hybrid_models(use_onnx=True)")
