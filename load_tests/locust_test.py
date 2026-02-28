import os
from io import BytesIO

from locust import HttpUser, between, task
from PIL import Image


class PredictUser(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self) -> None:
        image_path = os.getenv("TEST_IMAGE", "sample.jpg")
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                self.image_bytes = f.read()
        else:
            img = Image.new("RGB", (224, 224), color=(73, 109, 137))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            self.image_bytes = buf.getvalue()

    @task
    def predict(self) -> None:
        files = {"image": ("sample.jpg", self.image_bytes, "image/jpeg")}
        with self.client.post("/predict", files=files, catch_response=True, name="/predict") as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
            else:
                response.success()
