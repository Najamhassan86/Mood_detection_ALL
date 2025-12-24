"""Check UniFace's actual API"""
import inspect
from uniface import ArcFace, Emotion

print("=" * 60)
print("UNIFACE API CHECK")
print("=" * 60)

# Check Emotion
print("\n1. Emotion Model")
print("-" * 60)
emotion = Emotion()

print(f"Emotion Labels: {emotion.emotion_labels}")
print(f"Number of labels: {len(emotion.emotion_labels)}")

print("\nEmotion Public Methods:")
for name in dir(emotion):
    if not name.startswith('_'):
        attr = getattr(emotion, name)
        if callable(attr):
            sig = inspect.signature(attr) if hasattr(inspect, 'signature') else "..."
            print(f"  - {name}{sig}")

# Check ArcFace
print("\n2. ArcFace Model")
print("-" * 60)
arcface = ArcFace()

print("\nArcFace Public Methods:")
for name in dir(arcface):
    if not name.startswith('_'):
        attr = getattr(arcface, name)
        if callable(attr):
            try:
                sig = inspect.signature(attr)
                print(f"  - {name}{sig}")
            except:
                print(f"  - {name}(...)")

print("\n" + "=" * 60)