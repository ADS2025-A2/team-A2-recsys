import streamlit as st
import requests

key = "8983901b557993e61d5851a14d2d6c28"
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

@st.cache_data(show_spinner=False)
def get_movie_poster(title, year):
    search_url = f"{BASE_URL}/search/movie"

    params = {
        "api_key": key,
        "query": title,
        "year": year
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    if data["results"]:
        poster_path = data["results"][0].get("poster_path")
        if poster_path:
            return IMAGE_BASE + poster_path

    return None


@st.cache_data(show_spinner=False)
def get_movie_summary(title, year):
    search_url = f"{BASE_URL}/search/movie"
    params = {"api_key": key, "query": title, "year": year}
    response = requests.get(search_url, params=params)
    data = response.json()
    if data["results"]:
        summary = data["results"][0].get("overview")
        if summary:
            return summary
    return "Summary not available."


@st.cache_data(show_spinner=False)
def get_movie_cast(title, year):
    # First, search for the movie to get its TMDb ID
    search_url = f"{BASE_URL}/search/movie"
    params = {"api_key": key, "query": title, "year": year}
    response = requests.get(search_url, params=params)
    data = response.json()
    
    if not data["results"]:
        return ["Cast information not available."]
    
    movie_id = data["results"][0]["id"]
    
    # Now get movie credits
    credits_url = f"{BASE_URL}/movie/{movie_id}/credits"
    credits_params = {"api_key": key}
    credits_resp = requests.get(credits_url, params=credits_params).json()
    
    cast_list = [member["name"] for member in credits_resp.get("cast", [])[:5]]
    if cast_list:
        return cast_list
    else:
        return ["Cast information not available."]