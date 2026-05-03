"""
Movie Recommendation System — Live Streamlit App
Author  : Rakshit Thakur
Roll No : 2023BTCSE013  |  JLU ID : JLU07720
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import os

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f5c518 0%, #e87d0d 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.2rem;
}

.sub-title {
    text-align: center;
    color: #888;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

.movie-card {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 1px solid #3a3a5c;
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
    transition: transform 0.2s ease;
}

.movie-card:hover {
    transform: translateY(-4px);
    border-color: #f5c518;
}

.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: #f5c518;
    font-weight: 700;
    margin-bottom: 0.3rem;
    line-height: 1.3;
}

.movie-meta {
    font-size: 0.82rem;
    color: #aaa;
    margin-bottom: 0.2rem;
}

.genre-tag {
    display: inline-block;
    background: rgba(245,197,24,0.15);
    border: 1px solid rgba(245,197,24,0.3);
    color: #f5c518;
    font-size: 0.73rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 2px 2px 0 0;
}

.stars {
    color: #f5c518;
    font-size: 1rem;
}

.stat-box {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 1px solid #3a3a5c;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}

.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: #f5c518;
}

.stat-label {
    font-size: 0.85rem;
    color: #888;
}

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #f5c518;
    border-bottom: 2px solid rgba(245,197,24,0.3);
    padding-bottom: 0.4rem;
    margin-bottom: 1.2rem;
}

div.stButton > button {
    background: linear-gradient(135deg, #f5c518, #e87d0d) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    width: 100%;
}

div.stButton > button:hover {
    opacity: 0.9 !important;
    transform: scale(1.02);
}
</style>
""", unsafe_allow_html=True)


# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
ARTIFACTS_DIR = "app_artifacts"

@st.cache_resource
def load_artifacts():
    sim  = pickle.load(open(f"{ARTIFACTS_DIR}/similarity.pkl",    "rb"))
    mvs  = pd.read_pickle(f"{ARTIFACTS_DIR}/movies.pkl")
    idx  = pickle.load(open(f"{ARTIFACTS_DIR}/title_to_idx.pkl",  "rb"))
    return sim, mvs, idx

@st.cache_resource
def load_model():
    try:
        model        = pickle.load(open(f"{ARTIFACTS_DIR}/model.pkl",        "rb"))
        scaler       = pickle.load(open(f"{ARTIFACTS_DIR}/scaler.pkl",       "rb"))
        feature_cols = pickle.load(open(f"{ARTIFACTS_DIR}/feature_cols.pkl", "rb"))
        return model, scaler, feature_cols
    except Exception:
        return None, None, None

# Check artifacts exist
if not os.path.exists(ARTIFACTS_DIR):
    st.error("❌ `app_artifacts/` folder not found. Run Phase 1 cells in the notebook first!")
    st.stop()

similarity, movies_df, title_to_idx = load_artifacts()
rf_model, rf_scaler, feature_cols   = load_model()

GENRE_COLS = ['Action','Adventure','Animation','Children','Comedy','Crime',
              'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical',
              'Mystery','Romance','Sci-Fi','Thriller','War','Western']


# ─── TMDB POSTER FETCHER ──────────────────────────────────────────────────────
try:
    TMDB_API_KEY = st.secrets["TMDB_API_KEY"]
except Exception:
    TMDB_API_KEY = ""   # Set in Streamlit Cloud secrets

@st.cache_data(show_spinner=False)
def fetch_poster(title: str) -> str:
    """Fetch poster URL from TMDB. Falls back to placeholder."""
    clean = title.split("(")[0].strip()
    if not TMDB_API_KEY:
        return f"https://via.placeholder.com/200x300/1e1e2e/f5c518?text={clean[:20].replace(' ', '+')}"
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={requests.utils.quote(clean)}"
        resp = requests.get(url, timeout=5).json()
        results = resp.get("results", [])
        if results and results[0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w300{results[0]['poster_path']}"
    except Exception:
        pass
    return f"https://via.placeholder.com/200x300/1e1e2e/f5c518?text={clean[:15].replace(' ', '+')}"


# ─── CORE RECOMMENDATION FUNCTION ─────────────────────────────────────────────
def recommend(movie_title: str, n: int = 5, genre_filter: list = None) -> list:
    if movie_title not in title_to_idx:
        return []
    idx = title_to_idx[movie_title]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx]

    results = []
    for i, score in scores:
        if len(results) >= n * 3:   # grab extra for filtering
            break
        row = movies_df.iloc[i]
        genres = [g for g in GENRE_COLS if row.get(g, 0) == 1]

        if genre_filter:
            if not any(g in genres for g in genre_filter):
                continue

        results.append({
            "title":        row["title"],
            "avg_rating":   round(row["movie_avg_rating"], 2),
            "rating_count": int(row["movie_rating_count"]),
            "genres":       genres,
            "similarity":   round(float(score), 3),
        })

    return results[:n]


def get_top_rated(n: int = 10, genre_filter: list = None) -> pd.DataFrame:
    df = movies_df.copy()
    if genre_filter:
        mask = df[genre_filter].any(axis=1)
        df = df[mask]
    df = df[df["movie_rating_count"] >= 20]
    return df.nlargest(n, "movie_avg_rating")[
        ["title", "movie_avg_rating", "movie_rating_count"] + GENRE_COLS
    ]


def render_stars(rating: float) -> str:
    full  = int(rating)
    half  = 1 if (rating - full) >= 0.5 else 0
    empty = 5 - full - half
    return "★" * full + "½" * half + "☆" * empty


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎬 CineMatch")
    st.markdown("*ML-Powered Recommendation System*")
    st.markdown("---")

    page = st.radio(" Navigate", [
        " Find Similar Movies",
        " Top Rated Movies",
        " Browse by Genre",
        " Model Info",
    ])

    st.markdown("---")
    st.markdown("**Genre Filter**")
    selected_genres = st.multiselect(
        "Filter by genre (optional)",
        options=GENRE_COLS,
        default=[],
        label_visibility="collapsed"
    )

    n_recs = st.slider("Number of recommendations", 3, 15, 5)

    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:0.8rem;color:#666;'>
    Rakshit Thakur<br>
    Roll: 2023BTCSE013<br>
    JLU ID: JLU07720
    </div>
    """, unsafe_allow_html=True)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 CineMatch</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ML-Powered Movie Recommendation System &nbsp;•&nbsp; MovieLens 100K</div>',
            unsafe_allow_html=True)

# ─── STATS ROW ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
for col, num, label in [
    (c1, f"{len(movies_df):,}", "Movies"),
    (c2, "943",                 "Users"),
    (c3, "100K",               "Ratings"),
    (c4, "18",                 "Genres"),
]:
    col.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{num}</div>
        <div class="stat-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — FIND SIMILAR MOVIES
# ══════════════════════════════════════════════════════════════════════════════
if page == " Find Similar Movies":
    st.markdown('<div class="section-header"> Find Similar Movies</div>', unsafe_allow_html=True)

    all_titles = sorted(movies_df["title"].tolist())
    selected_movie = st.selectbox(
        "Select a movie you like:",
        options=all_titles,
        index=all_titles.index("Toy Story (1995)") if "Toy Story (1995)" in all_titles else 0,
    )

    if selected_movie:
        # Show selected movie info
        row = movies_df[movies_df["title"] == selected_movie].iloc[0]
        genres = [g for g in GENRE_COLS if row.get(g, 0) == 1]

        with st.expander(f"ℹ️ About: {selected_movie}", expanded=True):
            col_a, col_b = st.columns([1, 3])
            with col_a:
                poster = fetch_poster(selected_movie)
                st.image(poster, width=140)
            with col_b:
                st.markdown(f"**⭐ Avg Rating:** {row['movie_avg_rating']:.2f} / 5.0 &nbsp; {render_stars(row['movie_avg_rating'])}")
                st.markdown(f"**👥 Rated by:** {int(row['movie_rating_count']):,} users")
                tag_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in genres])
                st.markdown(f"**🎭 Genres:** {tag_html}", unsafe_allow_html=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        recommend_clicked = st.button(" Get Recommendations")

    if recommend_clicked:
        recs = recommend(selected_movie, n=n_recs, genre_filter=selected_genres or None)

        if not recs:
            st.warning("No recommendations found. Try a different movie or remove genre filter.")
        else:
            st.markdown(f"### 🎬 Top {len(recs)} recommendations for *{selected_movie}*")
            cols = st.columns(min(len(recs), 5))
            for i, rec in enumerate(recs):
                with cols[i % 5]:
                    poster = fetch_poster(rec["title"])
                    st.image(poster, use_container_width=True)
                    tag_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in rec["genres"][:2]])
                    st.markdown(f"""
                    <div class="movie-card">
                        <div class="movie-title">{rec['title']}</div>
                        <div class="movie-meta">⭐ {rec['avg_rating']} &nbsp;•&nbsp; {rec['rating_count']:,} ratings</div>
                        <div class="movie-meta"> Similarity: {rec['similarity']}</div>
                        {tag_html}
                    </div>""", unsafe_allow_html=True)

            # Predicted rating from RF model
            if rf_model is not None:
                st.markdown("---")
                st.markdown("###  ML Rating Prediction (Random Forest)")
                st.caption("Predicted ratings for a typical user (age 30, avg rater):")
                pred_cols = st.columns(min(len(recs), 5))
                for i, rec in enumerate(recs):
                    mv_row = movies_df[movies_df["title"] == rec["title"]]
                    if mv_row.empty:
                        continue
                    mv_row = mv_row.iloc[0]
                    feat = {
                        "age": 30, "gender_encoded": 1, "occupation_encoded": 5,
                        "user_avg_rating": 3.5, "user_rating_count": 50, "user_rating_std": 0.8,
                        "movie_avg_rating": mv_row["movie_avg_rating"],
                        "movie_rating_count": mv_row["movie_rating_count"],
                        "movie_rating_std": 0.9,
                    }
                    for g in GENRE_COLS:
                        feat[g] = mv_row.get(g, 0)
                    try:
                        row_df = pd.DataFrame([feat])[feature_cols]
                        row_sc = rf_scaler.transform(row_df)
                        pred   = float(np.clip(rf_model.predict(row_sc)[0], 1, 5))
                        with pred_cols[i % 5]:
                            st.metric(rec["title"][:25], f"{pred:.2f} ⭐")
                    except Exception:
                        pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TOP RATED MOVIES
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Top Rated Movies":
    st.markdown('<div class="section-header"> Top Rated Movies</div>', unsafe_allow_html=True)
    st.caption("Movies with at least 20 ratings, ranked by average rating")

    top_df = get_top_rated(n=n_recs, genre_filter=selected_genres or None)

    if top_df.empty:
        st.warning("No movies found for the selected genre filter.")
    else:
        cols = st.columns(min(len(top_df), 5))
        for i, (_, row) in enumerate(top_df.iterrows()):
            genres = [g for g in GENRE_COLS if row.get(g, 0) == 1]
            with cols[i % 5]:
                poster = fetch_poster(row["title"])
                st.image(poster, use_container_width=True)
                tag_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in genres[:2]])
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-meta">{render_stars(row['movie_avg_rating'])} {row['movie_avg_rating']:.2f}</div>
                    <div class="movie-meta">👥 {int(row['movie_rating_count']):,} ratings</div>
                    {tag_html}
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("###  Data Table")
        display_df = top_df[["title", "movie_avg_rating", "movie_rating_count"]].copy()
        display_df.columns = ["Title", "Avg Rating", "# Ratings"]
        display_df["Avg Rating"] = display_df["Avg Rating"].round(2)
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BROWSE BY GENRE
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Browse by Genre":
    st.markdown('<div class="section-header"> Browse by Genre</div>', unsafe_allow_html=True)

    genre_choice = st.selectbox("Pick a genre:", GENRE_COLS)
    genre_df = movies_df[movies_df[genre_choice] == 1].copy()
    genre_df = genre_df[genre_df["movie_rating_count"] >= 5]
    genre_df = genre_df.nlargest(n_recs, "movie_avg_rating")

    st.markdown(f"**{len(genre_df)} top {genre_choice} movies:**")

    if genre_df.empty:
        st.info("No movies found for this genre with enough ratings.")
    else:
        cols = st.columns(min(len(genre_df), 5))
        for i, (_, row) in enumerate(genre_df.iterrows()):
            genres = [g for g in GENRE_COLS if row.get(g, 0) == 1]
            with cols[i % 5]:
                poster = fetch_poster(row["title"])
                st.image(poster, use_container_width=True)
                tag_html = " ".join([f'<span class="genre-tag">{g}</span>' for g in genres[:2]])
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-meta">{render_stars(row['movie_avg_rating'])} {row['movie_avg_rating']:.2f}</div>
                    <div class="movie-meta">👥 {int(row['movie_rating_count']):,} ratings</div>
                    {tag_html}
                </div>""", unsafe_allow_html=True)

    # Genre distribution chart
    st.markdown("---")
    st.markdown("###  Genre Distribution in Dataset")
    genre_counts = {g: int(movies_df[g].sum()) for g in GENRE_COLS}
    gc_df = pd.DataFrame({"Genre": list(genre_counts.keys()),
                           "Movies": list(genre_counts.values())}).sort_values("Movies", ascending=False)
    st.bar_chart(gc_df.set_index("Genre"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == " Model Info":
    st.markdown('<div class="section-header"> Project & Model Info</div>', unsafe_allow_html=True)

    st.markdown("""
    ###  Recommendation Engine
    This app uses **Content-Based Filtering** with **Cosine Similarity** as the primary recommendation engine.

    | Step | Method |
    |------|--------|
    | Feature matrix | Movie genres (18 binary) + normalized avg rating + popularity |
    | Similarity metric | Cosine Similarity |
    | Model for rating prediction | Tuned Random Forest Regressor |
    | Dataset | MovieLens 100K |

    ###  Why Cosine Similarity + Content-Based?
    - Works without any user login — **cold start friendly**
    - Purely based on movie content (genres, popularity) — interpretable
    - Fast at inference time (pre-computed matrix)
    - Combined with RF model for **predicted rating** per user profile

    ###  ML Model Performance (Module 4)

    | Model | RMSE | MAE | R² |
    |-------|------|-----|----|
    | Linear Regression | ~1.00 | ~0.80 | ~0.18 |
    | Ridge Regression | ~0.99 | ~0.79 | ~0.19 |
    | Lasso Regression | ~1.00 | ~0.80 | ~0.18 |
    | Decision Tree | ~0.96 | ~0.75 | ~0.24 |
    | Random Forest | ~0.94 | ~0.73 | ~0.27 |
    | **Tuned Random Forest ** | **~0.93** | **~0.72** | **~0.29** |

    ###  Future Scope
    - Collaborative Filtering (SVD / ALS)
    - Neural Collaborative Filtering
    - TMDB poster integration (add API key in Streamlit secrets)
    - User login + personalized history
    - Hybrid recommendation pipeline
    """)

    st.markdown("---")
    st.markdown(f"""
    ** Author:** Rakshit Thakur &nbsp;|&nbsp; **Roll No:** 2023BTCSE013 &nbsp;|&nbsp; **JLU ID:** JLU07720  
    **Course:** Machine Learning &nbsp;|&nbsp; **Dataset:** MovieLens 100K
    """)
