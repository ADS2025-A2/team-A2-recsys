from fastapi import FastAPI

app = FastAPI()

# Esto evita errores circulares al importar las rutas
from .main import *
