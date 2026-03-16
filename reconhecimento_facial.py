import cv2
import numpy as np
import os
from deepface import DeepFace

PASTA_ROSTOS = "rostos_conhecidos"

def main():
    if not os.path.exists(PASTA_ROSTOS):
        os.makedirs(PASTA_ROSTOS)
        print(f"Pasta '{PASTA_ROSTOS}' criada. Adicione fotos e reinicie.")
        input("Pressione ENTER para sair...")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERRO] Nao foi possivel abrir a camera.")
        input("Pressione ENTER para sair...")
        return

    print("[INFO] Camera iniciada. Pressione Q para sair.")
    resultado_nome = "Analisando..."
    resultado_cor = (200, 200, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            resultado = DeepFace.find(
                img_path=frame,
                db_path=PASTA_ROSTOS,
                enforce_detection=False,
                silent=True
            )
            if len(resultado) > 0 and not resultado[0].empty:
                identidade = resultado[0].iloc[0]["identity"]
                nome = os.path.splitext(os.path.basename(identidade))[0]
                nome = nome.replace("_ok", "").replace("_", " ").title()
                resultado_nome = nome
                resultado_cor = (0, 220, 100)
            else:
                resultado_nome = "Desconhecido"
                resultado_cor = (0, 60, 220)
        except Exception:
            resultado_nome = "Nenhum rosto"
            resultado_cor = (100, 100, 100)

        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h-50), (w, h), (30, 30, 30), cv2.FILLED)
        cv2.putText(frame, resultado_nome, (10, h-15),
            cv2.FONT_HERSHEY_DUPLEX, 1.0, resultado_cor, 2)
        cv2.imshow("Reconhecimento Facial - Q para sair", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
