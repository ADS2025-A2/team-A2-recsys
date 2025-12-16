import sys

print("===== Verificando entorno de Python =====")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\n===== Verificando librerías =====")

required_packages = [
    "streamlit",
    "pandas",
    "numpy",
    "st_star_rating",  # tu módulo local
]

for pkg in required_packages:
    try:
        module = __import__(pkg)
        print(f"[OK] {pkg} -> {module.__file__}")
    except ImportError:
        print(f"[MISSING] {pkg} no está instalado")

print("\n===== Verificando módulo local streamlit_star_rating =====")
try:
    from streamlit_star_rating import st_star_rating
    print(f"[OK] st_star_rating -> {st_star_rating}")
except Exception as e:
    print(f"[ERROR] No se pudo importar st_star_rating: {e}")

print("\n===== Verificando Streamlit =====")
try:
    import streamlit as st
    print(f"[OK] Streamlit version: {st.__version__}")
except ImportError:
    print("[MISSING] Streamlit no está instalado")
