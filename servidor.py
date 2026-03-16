from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import os
DEEPFACE_IMPORT_ERROR = None

try:
    from deepface import DeepFace
except Exception as exc:
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = exc
import threading

app = Flask(__name__)
PASTA_ROSTOS = r"C:\Users\vf619\rostos_conhecidos"

indice_atual = 0
pessoas_detectadas = []
lock = threading.Lock()

# Camera dedicada so para exibir o video
cam_video = cv2.VideoCapture(0)
# Camera dedicada so para processar reconhecimento
cam_proc = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def processar_frames():
    if DeepFace is None:
        return

    global pessoas_detectadas
    frame_count = 0
    while True:
        success, frame = cam_proc.read()
        if not success:
            continue
        frame_count += 1
        if frame_count % 40 == 0:
            try:
                pequeno = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(pequeno, cv2.COLOR_BGR2GRAY)
                rostos = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40,40))
                img_h, img_w = frame.shape[:2]
                novas_pessoas = []
                for (x, y, w, h) in rostos:
                    x2, y2, w2, h2 = x*2, y*2, w*2, h*2
                    rosto_crop = frame[y2:y2+h2, x2:x2+w2]
                    if rosto_crop.size == 0:
                        continue
                    temp = f"tmp_{x}_{y}.jpg"
                    cv2.imwrite(temp, rosto_crop)
                    try:
                        busca = DeepFace.find(img_path=temp, db_path=PASTA_ROSTOS, enforce_detection=False, silent=True)
                        if len(busca) > 0 and not busca[0].empty:
                            identidade = busca[0].iloc[0]["identity"]
                            nome = os.path.splitext(os.path.basename(identidade))[0]
                            nome = nome.replace("_ok","").replace("_"," ").title()
                        else:
                            nome = "Desconhecido"
                    except:
                        nome = "Desconhecido"
                    if os.path.exists(temp):
                        os.remove(temp)
                    novas_pessoas.append({
                        "nome": nome,
                        "x": round(x2 / img_w * 100, 2),
                        "y": round(y2 / img_h * 100, 2),
                        "w": round(w2 / img_w * 100, 2),
                        "h": round(h2 / img_h * 100, 2),
                        "cor": "lime" if nome != "Desconhecido" else "red"
                    })
                with lock:
                    pessoas_detectadas = novas_pessoas
            except:
                pass

def gerar_frames():
    while True:
        success, frame = cam_video.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gerar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with lock:
        return jsonify(pessoas=pessoas_detectadas, camera=indice_atual)

@app.route('/trocar_camera', methods=['POST'])
def trocar_camera():
    global cam_video, cam_proc, indice_atual
    novo = 1 if indice_atual == 0 else 0
    nv = cv2.VideoCapture(novo)
    np2 = cv2.VideoCapture(novo)
    if nv.isOpened() and np2.isOpened():
        cam_video.release()
        cam_proc.release()
        cam_video = nv
        cam_proc = np2
        indice_atual = novo
        return jsonify(sucesso=True, camera=indice_atual)
    return jsonify(sucesso=False, camera=indice_atual)

if __name__ == '__main__':
    if DeepFace is None:
        print('[AVISO] DeepFace indisponivel. Reconhecimento desativado (video continua funcionando).')
        print('[AVISO] Para reconhecimento, use Python 3.10-3.12 com DeepFace/TensorFlow.')
        if DEEPFACE_IMPORT_ERROR is not None:
            print(f'[AVISO] Detalhe tecnico: {DEEPFACE_IMPORT_ERROR}')
    t = threading.Thread(target=processar_frames, daemon=True)
    t.start()
    print("\n[INFO] Acesse: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
