from __future__ import annotations

import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2

DEEPFACE_IMPORT_ERROR: Exception | None = None

try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = exc

BASE_DIR = Path(__file__).resolve().parent
KNOWN_FACES_DIR = BASE_DIR / "rostos_conhecidos"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def normalize_name(identity_path: str) -> str:
    return Path(identity_path).stem.replace("_ok", "").replace("_", " ").title()


def recognize_frame(frame_path: str) -> str:
    result = DeepFace.find(
        img_path=frame_path,
        db_path=str(KNOWN_FACES_DIR),
        enforce_detection=False,
        silent=True,
    )
    if len(result) > 0 and not result[0].empty:
        return normalize_name(result[0].iloc[0]["identity"])
    return "Desconhecido"


def main() -> None:
    configure_logging()

    if DeepFace is None:
        logging.error("DeepFace indisponivel: %s", DEEPFACE_IMPORT_ERROR)
        return

    KNOWN_FACES_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        logging.error("Nao foi possivel abrir a camera")
        return

    logging.info("Camera iniciada. Pressione 'q' para sair")
    frame_counter = 0
    last_name = "Analisando..."
    last_color = (200, 200, 0)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_counter += 1
        if frame_counter % 20 == 0:
            with NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, frame)

            try:
                last_name = recognize_frame(temp_path)
                last_color = (0, 220, 100) if last_name != "Desconhecido" else (0, 60, 220)
            except Exception as exc:  # pragma: no cover
                logging.debug("Erro no reconhecimento: %s", exc)
                last_name = "Nenhum rosto"
                last_color = (120, 120, 120)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, last_name, (10, h - 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, last_color, 2)
        cv2.imshow("Reconhecimento Facial - Q para sair", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
