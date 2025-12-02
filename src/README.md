# Team-A2 Recommender System Frontend

This repository contains a movie recommendation system with a **Streamlit frontend**.
This document explains how to set up the environment and run the **frontend**, working on both **macOS** and **Windows**.

---

## Prerequisites

* Python 3.10+ (tested with 3.10–3.13)
* pip
* Optional: virtual environment tool (`venv` or `virtualenv`)

---

## Setup Instructions

### 1. Create a virtual environment

#### Step 1.1: macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 1.2: Windows (Command Prompt)

```bash
python -m venv venv
venv\Scripts\activate
```

#### Step 1.3: Windows (PowerShell)

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

### 2. Fix broken virtual environments (if needed)

#### Step 2.1: Identify broken venv

If you get errors like:

```
/path/to/venv/bin/pip: line 2: .../venv/bin/python3: No such file or directory
```

then your virtual environment is broken.

#### Step 2.2: Fix broken venv

```bash
deactivate                 # exit current venv
rm -rf venv                # remove broken venv (Windows: rmdir /s /q venv)
python3 -m venv venv       # create a new venv
source venv/bin/activate   # activate it again (Windows adjust path as above)
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure you run this from the root of the repository, where `requirements.txt` is located.

---

### 3. Run the backend API (in a separate terminal)

The frontend depends on the backend API running locally.

#### Step 3.1: Open a new terminal

#### Step 3.2: Navigate to the backend folder

```bash
cd src/backend
```

#### Step 3.3: Run Uvicorn

```bash
uvicorn main:app --reload
```

* The API will run at [http://127.0.0.1:8000](http://127.0.0.1:8000)
* Keep this terminal open, as the frontend will make requests to it.

Important: The backend expects the MovieLens 10M dataset, make sure the files are in:

```
data/processed/ml-10M100K/
```

Must include at least: `movies.dat`, `ratings.dat`, `users.dat`.

---

### 4. Run the frontend (in a second terminal)

#### Step 4.1: Open another terminal

#### Step 4.2: Navigate to the frontend folder

```bash
cd src/app
```

#### Step 4.3: Run Streamlit

```bash
streamlit run Home.py
```

* Streamlit will open the frontend in your default browser.
* The frontend will fetch movies and recommendations from the backend running in the first terminal.

---

### 5. Notes

#### Step 5.1: Recommended movies 

* Loaded from `training/top10_recommendations_with_titles.csv`.

#### Step 5.2: Average rating per movie and unique genres

* Loaded from `data/processed/avg_rating_per_movie.csv`.
* Loaded from `data/processed/unique_genres.csv`.

#### Step 5.3: Usage

* Use the login/registration form in Streamlit; cookies manage sessions.
* Search for movies using the multiselect box or click recommended movies to see details.
* Selecting multiple movies in the dropdown will show details for all selected movies.

---

### 6. Troubleshooting

#### Step 6.1: ModuleNotFoundError

* Ensure your venv is activated.
* Run `pip install -r requirements.txt` from the root of the repository.

#### Step 6.2: FileNotFoundError for CSVs or MovieLens .dat files

* Check that the folder structure matches the expected one.
* Make sure the dataset files exist and names are correct.

#### Step 6.3: Broken virtual environment

* Follow the **Fix broken virtual environments** section.

#### Step 6.4: Backend not reachable

* Make sure you have two terminals open: one running the backend and one the frontend.
* Backend must run before opening the frontend.

#### Step 6.5: Port conflicts

* Streamlit default: 8501, Uvicorn default: 8000
* Change ports if needed:

```bash
streamlit run Home.py --server.port 8502
uvicorn main:app --port 8001 --reload
```

---
