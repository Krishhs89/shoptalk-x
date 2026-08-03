"""
Locust load test for the ShopTalk-X API (design doc §4.2, "measure and
report P50/P95/P99 per stage"). Exercises /search/text, /search/image, and
/health at realistic ratios.

Usage:
  locust -f src/shoptalk/loadtest/locustfile.py --host http://localhost:8000 \\
    --users 10 --spawn-rate 2 --run-time 2m --headless \\
    --csv results/day6_locust
"""
import random
from pathlib import Path

from locust import HttpUser, between, task

QUERIES = [
    "red shirt for men under 50 dollars",
    "colorful phone case",
    "comfortable running shoes",
    "wireless headphones",
    "kitchen storage containers",
    "leather handbag",
    "affordable desk lamp",
    "waterproof backpack",
]

SAMPLE_IMAGE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "images"


class ShopTalkUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        self.session_id = None
        images = list(SAMPLE_IMAGE_DIR.glob("*.jpg")) if SAMPLE_IMAGE_DIR.exists() else []
        self.sample_image = images[0] if images else None

    @task(6)
    def search_text(self):
        payload = {"query": random.choice(QUERIES), "session_id": self.session_id, "stream": False}
        with self.client.post("/search/text", json=payload, catch_response=True, name="/search/text") as resp:
            if resp.ok:
                self.session_id = resp.json().get("session_id")
            else:
                resp.failure(f"status {resp.status_code}")

    @task(2)
    def search_image(self):
        if not self.sample_image:
            return
        with open(self.sample_image, "rb") as f:
            files = {"file": (self.sample_image.name, f, "image/jpeg")}
            with self.client.post("/search/image", files=files, name="/search/image", catch_response=True) as resp:
                if not resp.ok:
                    resp.failure(f"status {resp.status_code}")

    @task(3)
    def health(self):
        self.client.get("/health", name="/health")
