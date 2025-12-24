import cv2
import numpy as np
import pickle
import os
from pathlib import Path
import time
from typing import Dict, Tuple

# UniFace
from uniface import RetinaFace, ArcFace, compute_similarity
from uniface.constants import RetinaFaceWeights


class FaceRecognitionSystem:

    def __init__(
        self,
        database_path="database",
        embeddings_path="embeddings.pkl",
        similarity_threshold=0.4
    ):
        self.database_path = Path(database_path)
        self.embeddings_path = Path(embeddings_path)
        self.similarity_threshold = similarity_threshold

        print("🔄 Loading models...")
        self.detector = RetinaFace(
            model_name=RetinaFaceWeights.MNET_V2,
            conf_thresh=0.6,
            input_size=(640, 640)
        )
        self.recognizer = ArcFace()
        self.face_database: Dict[str, np.ndarray] = {}
        print("✅ Models loaded\n")

    # --------------------------------------------------
    # REGISTRATION
    # --------------------------------------------------
    def register_new_person(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("❌ Webcam not accessible")
            return

        poses = ["STRAIGHT", "LEFT", "RIGHT", "UP", "DOWN"]
        pose_index = 0
        captured = []
        last_capture_time = 0

        print("\n📸 Starting Face Registration")
        print("➡ Follow on-screen instructions")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detector.detect(frame)

            if len(faces) != 1:
                cv2.putText(frame, "Ensure ONLY ONE face is visible",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 255), 2)
                cv2.imshow("Registration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            face = faces[0]
            landmarks = face["landmarks"]

            # Pose estimation (simple heuristic)
            left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]

            dx = nose[0] - (left_eye[0] + right_eye[0]) / 2
            dy = nose[1] - (left_eye[1] + right_eye[1]) / 2

            pose_ok = False
            current_pose = poses[pose_index]

            if current_pose == "STRAIGHT" and abs(dx) < 10 and abs(dy) < 10:
                pose_ok = True
            elif current_pose == "LEFT" and dx > 15:
                pose_ok = True
            elif current_pose == "RIGHT" and dx < -15:
                pose_ok = True
            elif current_pose == "UP" and dy < -15:
                pose_ok = True
            elif current_pose == "DOWN" and dy > 15:
                pose_ok = True

            if pose_ok and time.time() - last_capture_time > 0.8:
                captured.append(frame.copy())
                last_capture_time = time.time()
                pose_index += 1
                print(f"✅ Captured pose: {current_pose}")

            # Draw UI
            cv2.rectangle(frame,
                          (int(face["bbox"][0]), int(face["bbox"][1])),
                          (int(face["bbox"][2]), int(face["bbox"][3])),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"POSE: {current_pose}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

            cv2.putText(frame,
                        f"Progress: {pose_index}/{len(poses)}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2)

            cv2.imshow("Registration", frame)

            if pose_index >= len(poses):
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if len(captured) < len(poses):
            print("❌ Registration incomplete")
            return

        name = input("\nEnter person's name/ID: ").strip()
        person_dir = self.database_path / name

        if person_dir.exists():
            overwrite = input("⚠ Person exists. Overwrite? (y/n): ")
            if overwrite.lower() != "y":
                print("❌ Registration cancelled")
                return
        person_dir.mkdir(parents=True, exist_ok=True)

        embeddings = []

        for i, img in enumerate(captured):
            img_path = person_dir / f"reg_{i+1:02d}.jpg"
            cv2.imwrite(str(img_path), img)

            faces = self.detector.detect(img)
            if faces:
                emb = self.recognizer.get_normalized_embedding(
                    img, faces[0]["landmarks"]
                )
                embeddings.append(emb)

        if embeddings:
            self.face_database[name] = np.mean(embeddings, axis=0)
            self._save_database()
            print(f"✅ {name} registered successfully!\n")
        else:
            print("❌ Failed to generate embeddings")

    # --------------------------------------------------
    # DATABASE
    # --------------------------------------------------
    def _save_database(self):
        with open(self.embeddings_path, "wb") as f:
            pickle.dump(self.face_database, f)

    def load_database(self):
        if not self.embeddings_path.exists():
            print("❌ No database found")
            return False
        with open(self.embeddings_path, "rb") as f:
            self.face_database = pickle.load(f)
        print(f"✅ Loaded {len(self.face_database)} people")
        return True

    def recognize_face(self, embedding) -> Tuple[str, float]:
        best_name = "Unknown"
        best_score = 0.0

        for name, db_emb in self.face_database.items():
            sim = compute_similarity(embedding, db_emb)

            # 🔒 Normalize similarity output safely
            if isinstance(sim, np.ndarray):
                sim = float(sim.flatten()[0])
            else:
                sim = float(sim)

            if sim > best_score:
                best_score = sim
                best_name = name

        if best_score < self.similarity_threshold:
            return "Unknown", best_score

        return best_name, best_score


    # --------------------------------------------------
    # WEBCAM RECOGNITION
    # --------------------------------------------------
    def run_webcam_recognition(self):
        if not self.face_database:
            print("❌ Load database first")
            return

        cap = cv2.VideoCapture(0)
        print("\n🎥 Recognition started | Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.detector.detect(frame)
            for face in faces:
                emb = self.recognizer.get_normalized_embedding(
                    frame, face["landmarks"]
                )
                name, score = self.recognize_face(emb)

                x1, y1, x2, y2 = map(int, face["bbox"])
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(frame,
                            f"{name} ({score:.2f})",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, color, 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# --------------------------------------------------
# MAIN MENU
# --------------------------------------------------
def main():
    system = FaceRecognitionSystem()

    while True:
        print("\n" + "=" * 60)
        print("📋 MENU")
        print("1. Register New Person (Webcam)")
        print("2. Load Existing Database")
        print("3. Start Webcam Recognition")
        print("4. Exit")
        print("=" * 60)

        choice = input("Select option: ").strip()

        if choice == "1":
            system.register_new_person()
        elif choice == "2":
            system.load_database()
        elif choice == "3":
            system.run_webcam_recognition()
        elif choice == "4":
            print("👋 Bye")
            break
        else:
            print("❌ Invalid option")


if __name__ == "__main__":
    main()
