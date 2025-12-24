import cv2

from uniface import RetinaFace, Emotion
from uniface.visualization import draw_detections


def draw_emotion_label(image, bbox, emotion: str, confidence: float):
    x1, y1 = int(bbox[0]), int(bbox[1])
    text = f'{emotion} ({confidence:.2f})'
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

    cv2.rectangle(
        image,
        (x1, y1 - th - 10),
        (x1 + tw + 10, y1),
        (255, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x1 + 5, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def main():
    # ✅ OFFICIAL INITIALIZATION (NO ARGS)
    detector = RetinaFace()
    emotion_predictor = Emotion()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    print("🎭 UniFace Emotion Detection (press 'q' to quit)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        faces = detector.detect(frame)

        # visualize detections
        bboxes = [f["bbox"] for f in faces]
        scores = [f["confidence"] for f in faces]
        landmarks = [f["landmarks"] for f in faces]

        draw_detections(
            image=frame,
            bboxes=bboxes,
            scores=scores,
            landmarks=landmarks,
            vis_threshold=0.6,
            fancy_bbox=True,
        )

        # emotion prediction (✅ LANDMARK-BASED API)
        for face in faces:
            emotion, confidence = emotion_predictor.predict(
                frame, face["landmarks"]
            )
            draw_emotion_label(
                frame, face["bbox"], emotion, confidence
            )

        cv2.putText(
            frame,
            f"Faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
