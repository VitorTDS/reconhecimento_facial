# Sistema Profissional de Reconhecimento Facial

Aplicação web em Flask com streaming em tempo real, painel operacional e motor de reconhecimento desacoplado em thread dedicada.

## Destaques

- **Dashboard profissional** com indicadores de saúde, pessoas detectadas e overlays de bounding box.
- **Engine separada** para streaming e processamento facial, reduzindo bloqueios da interface.
- **Endpoint de saúde (`/health`)** para monitoramento e observabilidade.
- **Fallback robusto**: sistema continua disponível em modo streaming mesmo sem DeepFace/TensorFlow.
- **Compatível com deployment WSGI** via Gunicorn.

## Requisitos

- Python **3.10 a 3.12** para reconhecimento completo com DeepFace/TensorFlow.
- Webcam local.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

### Desenvolvimento

```bash
python servidor.py
```

### Produção (exemplo)

```bash
gunicorn -w 2 -b 0.0.0.0:5000 servidor:app
```

## Endpoints

- `GET /`: dashboard.
- `GET /video`: stream MJPEG.
- `GET /status`: snapshot operacional (câmera, pessoas, base).
- `POST /trocar_camera`: alterna entre índice 0 e 1.
- `GET /health`: status básico da aplicação.

## Estrutura

- `servidor.py`: app Flask + classe `RecognitionEngine`.
- `reconhecimento_facial.py`: modo standalone em janela OpenCV.
- `templates/index.html`: dashboard e frontend.
- `teste.py`: smoke check de dependências.
