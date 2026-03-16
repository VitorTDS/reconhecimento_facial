# Reconhecimento Facial

Este projeto usa OpenCV + DeepFace para reconhecer rostos pela câmera.

## 1) Criar ambiente virtual (Windows / PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

> Se você não tiver Python 3.10 instalado, instale Python 3.10 ou 3.11.
> Versões muito novas do Python podem não ter wheel compatível para TensorFlow em alguns cenários.

## 2) Instalar dependências

```powershell
pip install -r requerimentos.txt
```

## 3) Rodar o script principal

```powershell
python reconhecimento_facial.py
```

## Erro comum no Windows: `No matching distribution found for tensorflow...`

Isso normalmente acontece por incompatibilidade entre a versão do Python e o pacote TensorFlow disponível.

Faça este checklist:

1. Confira sua versão do Python:
   ```powershell
   python --version
   ```
2. Se estiver muito nova (ex.: 3.12+ em alguns ambientes), recrie o venv com Python 3.10/3.11.
3. Atualize o pip antes de instalar:
   ```powershell
   python -m pip install --upgrade pip setuptools wheel
   ```
4. Tente instalar novamente:
   ```powershell
   pip install -r requerimentos.txt
   ```

## Dependências

O arquivo `requerimentos.txt` já contém:

- `opencv-python`
- `numpy`
- `deepface`
- `tensorflow-intel` para Windows
- `tensorflow` para Linux/macOS
