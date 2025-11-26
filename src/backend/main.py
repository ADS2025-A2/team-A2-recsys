# backend/main.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import os
import sys

# ---------------------------
# Importar recomendaciones dummy
# ---------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "App", "model")))
from recommendations import DUMMY_RECOMMENDATIONS

app = FastAPI()

# ---------------------------
# Cargar movies.dat
# ---------------------------
movies_path = "../data/processed/ml-10M100K/movies.dat"
movies_df = pd.read_csv(
    movies_path,
    sep=",",   # tu separador real
    header=0,  # ya tiene header: movie_id,title,genres
    names=["movie_id", "title", "genres"]
)

# ---------------------------
# Endpoint para catálogo de películas
# ---------------------------
@app.get("/movies")
def get_movies():
    try:
        movies_list = movies_df[["title", "genres"]].to_dict(orient="records")
        return movies_list
    except Exception as e:
        print("ERROR en /movies:", e)
        return {"error": str(e)}

# ---------------------------
# Endpoint para recomendaciones por usuario
# ---------------------------
@app.get("/api/recommendations")
def api_recommendations(user_id: str):
    """
    Retorna las top 10 películas recomendadas para un usuario dado.
    user_id puede ser número o string (username).
    """
    try:
        # convertir username a int simple para usar el diccionario
        try:
            numeric_id = int(user_id)
        except:
            numeric_id = sum([ord(c) for c in user_id]) % 10 + 1  # solo para test

        recommendations = DUMMY_RECOMMENDATIONS.get(numeric_id, [])
        return JSONResponse(content={"recommendations": recommendations})

    except Exception as e:
        print("ERROR en /api/recommendations:", e)
        return JSONResponse(content={"recommendations": []})
