from __future__ import annotations

import os
import time
from typing import Generator, Optional

import cv2
from flask import Flask, Response, render_template

from plate_pipeline.core import detect_recognition_plate, draw_result, load_models


def create_app(
    detect_weights: str = "weights/plate_detect.pt",
    rec_weights: str = "weights/plate_rec_color.pth",
    is_color: bool = True,
    camera_index: int = 0,
    save_dir: Optional[str] = None,
) -> Flask:
    app = Flask(__name__)

    models = load_models(detect_weights=detect_weights, rec_weights=rec_weights, is_color=is_color)
    save_dir = save_dir or os.environ.get("PLATE_SAVE_DIR") or "photos"
    os.makedirs(save_dir, exist_ok=True)

    def generate_frames() -> Generator[bytes, None, None]:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_interval = int(fps * 3)
        frame_count = 0
        image_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break

            dict_list = detect_recognition_plate(
                models.detect_model, frame, models.device, models.plate_rec_model, img_size=640, is_color=is_color
            )
            frame = draw_result(frame, dict_list)

            if frame_count % max(1, frame_interval) == 0:
                image_path = os.path.join(save_dir, f"frame_{image_count}.jpg")
                cv2.imwrite(image_path, frame)
                image_count += 1

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            frame_count += 1

        cap.release()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main() -> None:
    host = os.environ.get("PLATE_FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("PLATE_FLASK_PORT", "5000"))
    app = create_app()
    app.run(host=host, port=port)


if __name__ == "__main__":
    main()

