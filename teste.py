"""Smoke checks for local environment dependencies."""

from importlib import import_module

MODULES = ["cv2", "numpy", "flask"]
OPTIONAL = ["deepface"]

for module in MODULES:
    import_module(module)
    print(f"[OK] modulo '{module}' carregado")

for module in OPTIONAL:
    try:
        import_module(module)
        print(f"[OK] opcional '{module}' carregado")
    except Exception as exc:
        print(f"[WARN] opcional '{module}' indisponivel: {exc}")
