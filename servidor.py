from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

DEEPFACE_IMPORT_ERROR: Exception | None = None

try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover - ambiente pode nao suportar tensorflow/deepface
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = exc


@dataclass
class FaceDetection:
    nome: str
    x: float
    y: float
    w: float
    h: float
    cor: str


class RecognitionEngine:
    """Gerencia captura de video e reconhecimento em thread separada."""

    def __init__(self, known_faces_dir: Path, frame_skip: int = 25, jpeg_quality: int = 75) -> None:
        self.known_faces_dir = known_faces_dir
        self.frame_skip = frame_skip
        self.jpeg_quality = jpeg_quality
        self.camera_index = 0
        self._lock = threading.Lock()
        self._detections: list[FaceDetection] = []
        self._running = False
        self._processing_thread: threading.Thread | None = None
        self._video_cap: cv2.VideoCapture | None = None
        self._proc_cap: cv2.VideoCapture | None = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def start(self) -> None:
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        self._video_cap = cv2.VideoCapture(self.camera_index)
        self._proc_cap = cv2.VideoCapture(self.camera_index)

        self._running = True
        self._processing_thread = threading.Thread(
            target=self._process_frames, name="recognition-worker", daemon=True
        )
        self._processing_thread.start()
        logging.info("RecognitionEngine iniciado (camera=%s)", self.camera_index)

    def shutdown(self) -> None:
        self._running = False
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2)
        if self._video_cap is not None:
            self._video_cap.release()
        if self._proc_cap is not None:
            self._proc_cap.release()
        logging.info("RecognitionEngine finalizado")

    def _read_frame(self, capture: cv2.VideoCapture | None) -> tuple[bool, Any]:
        if capture is None or not capture.isOpened():
            return False, None
        return capture.read()

    def stream_frames(self):
        while self._running:
            ok, frame = self._read_frame(self._video_cap)
            if not ok:
                frame = self._build_fallback_frame("Camera indisponivel")
                time.sleep(0.2)

            ret, buffer = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ret:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"

    def switch_camera(self) -> bool:
        target_index = 1 if self.camera_index == 0 else 0
        new_video = cv2.VideoCapture(target_index)
        new_proc = cv2.VideoCapture(target_index)
        if not (new_video.isOpened() and new_proc.isOpened()):
            new_video.release()
            new_proc.release()
            return False

        if self._video_cap is not None:
            self._video_cap.release()
        if self._proc_cap is not None:
            self._proc_cap.release()

        self._video_cap = new_video
        self._proc_cap = new_proc
        self.camera_index = target_index
        logging.info("Camera alternada para indice %s", self.camera_index)
        return True

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            pessoas = [d.__dict__ for d in self._detections]
        return {
            "camera": self.camera_index,
            "pessoas": pessoas,
            "engine": {
                "deepface_enabled": DeepFace is not None,
                "known_faces": len(list(self.known_faces_dir.glob("*"))),
            },
        }

    def _process_frames(self) -> None:
        if DeepFace is None:
            logging.warning("DeepFace indisponivel; modo apenas stream de video.")
            return

        frame_counter = 0
        while self._running:
            ok, frame = self._read_frame(self._proc_cap)
            if not ok:
                time.sleep(0.2)
                continue

            frame_counter += 1
            if frame_counter % self.frame_skip != 0:
                continue

            detections = self._detect_faces(frame)
            with self._lock:
                self._detections = detections

    def _detect_faces(self, frame: Any) -> list[FaceDetection]:
        preview = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        img_h, img_w = frame.shape[:2]
        detected: list[FaceDetection] = []

        for (x, y, w, h) in faces:
            x2, y2, w2, h2 = x * 2, y * 2, w * 2, h * 2
            face_crop = frame[y2 : y2 + h2, x2 : x2 + w2]
            if face_crop.size == 0:
                continue

            name = self._identify_face(face_crop)
            detected.append(
                FaceDetection(
                    nome=name,
                    x=round((x2 / img_w) * 100, 2),
                    y=round((y2 / img_h) * 100, 2),
                    w=round((w2 / img_w) * 100, 2),
                    h=round((h2 / img_h) * 100, 2),
                    cor="lime" if name != "Desconhecido" else "red",
                )
            )
        return detected

    def _identify_face(self, face_crop: Any) -> str:
        with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, face_crop)

        try:
            result = DeepFace.find(
                img_path=tmp_path,
                db_path=str(self.known_faces_dir),
                enforce_detection=False,
                silent=True,
            )
            if len(result) > 0 and not result[0].empty:
                identity = result[0].iloc[0]["identity"]
                return (
                    Path(identity)
                    .stem.replace("_ok", "")
                    .replace("_", " ")
                    .title()
                )
            return "Desconhecido"
        except Exception as exc:  # pragma: no cover - robustez em runtime
            logging.debug("Falha ao identificar rosto: %s", exc)
            return "Desconhecido"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _build_fallback_frame(message: str) -> Any:
        canvas = np.zeros((480, 640, 3), dtype="uint8")
        cv2.putText(
            canvas,
            message,
            (90, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return canvas


def create_app() -> Flask:
    base_dir = Path(__file__).resolve().parent
    known_faces_dir = base_dir / "rostos_conhecidos"

    logging.basicConfig(
        level=os.getenv("RF_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
    )

    app = Flask(__name__)
    engine = RecognitionEngine(known_faces_dir=known_faces_dir)
    app.config["ENGINE"] = engine

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video")
    def video():
        return Response(
            engine.stream_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/status")
    def status():
        return jsonify(engine.status_payload())

    @app.post("/trocar_camera")
    def trocar_camera():
        success = engine.switch_camera()
        return jsonify(sucesso=success, camera=engine.camera_index)

    @app.get("/health")
    def health():
        return jsonify(status="ok", deepface=DeepFace is not None)

    if DeepFace is None:
        logging.warning("DeepFace indisponivel: %s", DEEPFACE_IMPORT_ERROR)

    engine.start()
    atexit.register(engine.shutdown)
    return app


app = create_app()


if __name__ == "__main__":
    logging.info("Servidor disponivel em http://localhost:5000")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
