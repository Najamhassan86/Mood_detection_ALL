import time
import random
from datetime import datetime, timezone

import requests

# ==============================
# CONFIG
# ==============================

BACKEND_BASE = "https://emotion-backend-production.up.railway.app"  # <-- change this
DEVICE_ID = "jetson_1"

BATCH_ENDPOINT = f"{BACKEND_BASE}/api/emotions/batch"


def build_dummy_payload():
    """
    Build a fake batch payload that looks like what
    the real Jetson loop will send later.
    """

    now = datetime.now(timezone.utc).isoformat()

    # Two fake people with changing cumulative times
    # In a real system, these would be your tracked persons.
    person_1_happy = random.uniform(50.0, 200.0)
    person_1_sad = random.uniform(0.0, 50.0)
    person_1_angry = random.uniform(0.0, 20.0)

    person_2_happy = random.uniform(0.0, 80.0)
    person_2_sad = random.uniform(20.0, 150.0)
    person_2_angry = random.uniform(0.0, 40.0)

    payload = {
        "device_id": DEVICE_ID,
        "timestamp": now,
        "people": [
            {
                "person_id": "1",
                "cumulative": {
                    "happy": person_1_happy,
                    "sad": person_1_sad,
                    "angry": person_1_angry,
                },
            },
            {
                "person_id": "2",
                "cumulative": {
                    "happy": person_2_happy,
                    "sad": person_2_sad,
                    "angry": person_2_angry,
                },
            },
        ],
    }

    return payload


def main():
    print(f"Sending dummy emotion batches to: {BATCH_ENDPOINT}")
    print("Press Ctrl+C to stop.\n")

    while True:
        payload = build_dummy_payload()
        try:
            resp = requests.post(BATCH_ENDPOINT, json=payload, timeout=5)
            print(
                f"[{datetime.now().isoformat()}] "
                f"Status: {resp.status_code}  Sent {len(payload['people'])} people"
            )
            if resp.status_code != 200:
                print("  Response text:", resp.text)
        except Exception as e:
            print(f"Error sending batch: {e}")

        # wait a few seconds before sending next batch
        time.sleep(5)


if __name__ == "__main__":
    main()
