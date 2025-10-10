"""Streamlit entry point. Run with: python -m streamlit run app.py."""
from collections import Counter
import csv
from html import escape
from pathlib import Path
import re
import os

import streamlit as st

ROOT = Path(__file__).resolve().parent
SKIPPED = "Haven’t Seen"
RATING_OPTIONS = [SKIPPED] + [value / 2 for value in range(1, 11)]
RATING_TARGET = 5


@st.cache_resource(show_spinner=False)
def load_catalog():
    with (ROOT / "data/processed/movies_processed.csv").open(newline="", encoding="utf-8") as file:
        movies = {int(row["movieId"]): (row["title"], row["genres"])
                  for row in csv.DictReader(file)}
    if not movies:
        raise ValueError("The movie catalog is empty.")
    genres = sorted({genre for _, listed in movies.values() for genre in listed.split("|")
                     if genre and genre != "(no genres listed)"})
    title_counts = Counter(title for title, _ in movies.values())
    labels = {movie_id: f"{title} · Movie {movie_id}" if title_counts[title] > 1 else title
              for movie_id, (title, _) in movies.items()}
    movie_ids = sorted(movies, key=lambda movie_id: labels[movie_id].casefold())
    return movies, genres, labels, movie_ids


@st.cache_resource(show_spinner="Preparing your recommendations…")
def load_recommender(use_embeddings):
    from src.hybrid_recommender import HybridRecommender

    return HybridRecommender.from_project(ROOT, use_embeddings=use_embeddings)


def poster_credentials():
    """Environment variables take priority over optional local Streamlit secrets."""
    values = []
    for name in ["TMDB_API_KEY", "TMDB_READ_ACCESS_TOKEN"]:
        value = os.environ.get(name, "").strip()
        if not value:
            try:
                value = str(st.secrets.get(name, "")).strip()
            except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
                value = ""
        values.append(value)
    return tuple(values)


@st.cache_data(show_spinner=False)
def poster_mapping():
    from src.posters import load_tmdb_mapping

    return load_tmdb_mapping(ROOT / "data/raw/links.csv")


@st.cache_data(ttl=3600, max_entries=1000, show_spinner=False)
def cached_poster(tmdb_id, api_key, read_token):
    from src.posters import fetch_poster

    # Include credentials in the cache key so configuring/changing a key retries.
    return fetch_poster(tmdb_id, api_key, read_token)


def movie_poster(movie_id):
    api_key, read_token = poster_credentials()
    if not api_key and not read_token:
        return None
    tmdb_id = poster_mapping().get(int(movie_id))
    return cached_poster(tmdb_id, api_key, read_token) if tmdb_id else None


def cover_html(movie_id, title, rank=None):
    poster = movie_poster(movie_id)
    cover_class = "search-cover" if rank is None else "movie-cover"
    rank_badge = "" if rank is None else f'<span class="rank">{rank:02d}</span>'
    if poster:
        return (f'<div class="{cover_class} has-poster">'
                f'<img src="{poster}" alt="{escape(title, quote=True)} poster" />'
                f'{rank_badge}</div>')
    if rank is None:
        return f'<div class="{cover_class} palette-{int(movie_id) % 6}">Poster unavailable</div>'
    return (f'<div class="{cover_class} palette-{int(movie_id) % 6}">'
            f'{rank_badge}<span class="cover-mark" aria-hidden="true">▶</span>'
            '<span class="cover-label">Poster unavailable</span></div>')


def go_to(step):
    st.session_state.step = step


def choose_mode(mode):
    if st.session_state.discovery_mode != mode:
        st.session_state.pop("recommendation_result", None)
        st.session_state.favorite_genres = []
        st.session_state.pop("genre_selection", None)
    st.session_state.discovery_mode = mode
    go_to(2 if mode == "movies" else 1)


def selected_genres():
    return tuple(st.session_state.favorite_genres) if st.session_state.discovery_mode == "genres" else ()


def save_genres():
    st.session_state.favorite_genres = sorted(st.session_state.genre_selection)


def save_rating(movie_id, widget_key):
    value = st.session_state[widget_key]
    ratings = st.session_state.user_ratings
    skipped = st.session_state.skipped_movies
    if value == SKIPPED:
        ratings.pop(movie_id, None)
        skipped.add(movie_id)
    else:
        skipped.discard(movie_id)
        if value is None:
            ratings.pop(movie_id, None)
        else:
            ratings[movie_id] = value
    other_prefix = "rate" if widget_key.startswith("edit_") else "edit"
    st.session_state.pop(f"{other_prefix}_{movie_id}", None)


def remove_rating(movie_id):
    st.session_state.user_ratings.pop(movie_id, None)
    for prefix in ("rate", "edit"):
        st.session_state.pop(f"{prefix}_{movie_id}", None)


def title_parts(title):
    match = re.search(r"\s*\((\d{4})\)\s*$", title)
    return (title[:match.start()], match.group(1)) if match else (title, "Year unavailable")


def genre_label(genres):
    return "Genres unavailable" if genres == "(no genres listed)" else genres.replace("|", " · ")


def rating_label(value):
    if value == SKIPPED:
        return SKIPPED
    return f"{'★' * int(value)}{'½' if value % 1 else ''}  {value:g} / 5"


def header(step, mode):
    st.html('''<header class="brand-bar"><div class="brand"><span class="brand-icon">▶</span>
    <div><strong>PERSONALIZED</strong><span>Movie Recommender</span></div></div>
    <span class="brand-tag">Your next favorite is waiting</span></header>''')
    if step == 0:
        return
    stages = [(1, "Preferences"), (2, "Rate Movies"), (3, "Recommendations")]
    if mode == "movies":
        stages = stages[1:]
    items = []
    for index, (stage, label) in enumerate(stages, 1):
        state = "complete" if stage < step else "active" if stage == step else "future"
        items.append(f'<div class="step {state}"><b>{"✓" if stage < step else index}</b><span>{label}</span></div>')
    st.html('<nav aria-label="Recommendation progress" class="stepper">' + '<i></i>'.join(items) + '</nav>')


def discovery():
    st.html('''<section class="hero"><p class="eyebrow">CURATED FOR YOU</p>
    <h1>How would you like to <span>discover movies</span>?</h1></section>''')
    movies, genres = st.columns(2, gap="large")
    with movies, st.container(border=True, key="discovery_movies"):
        st.subheader("Based on movies I've watched")
        st.write("Search for and rate five movies you’ve seen. Your ratings alone guide your recommendations.")
        st.button("Start with movie ratings", key="choose_movies", type="primary", width="stretch",
                  on_click=choose_mode, args=("movies",))
    with genres, st.container(border=True, key="discovery_genres"):
        st.subheader("Based on my favorite genres")
        st.write("Choose the genres you enjoy. If you add movie ratings, they’ll have more influence on your picks.")
        st.button("Start with favorite genres", key="choose_genres", type="primary", width="stretch",
                  on_click=choose_mode, args=("genres",))


@st.fragment
def genre_picker(genres):
    with st.container(border=True, key="genre_panel"):
        st.subheader("What do you love to watch?")
        st.caption("Select as many as you like.")
        selected = st.pills("Favorite genres", genres, selection_mode="multi",
                            default=st.session_state.favorite_genres, key="genre_selection",
                            on_change=save_genres, label_visibility="collapsed", wrap=True)
        st.html(f'<span class="selection-count" role="status">{len(selected)} selected</span>')
    back, next_column = st.columns([3, 1])
    if back.button("Change discovery method", key="change_mode"):
        go_to(0)
        st.rerun()
    if next_column.button("Continue  →", type="primary", width="stretch", key="continue",
                          disabled=not selected):
        go_to(2)
        st.rerun()


def preferences(genres):
    st.html('''<section class="hero"><p class="eyebrow">CURATED FOR YOU</p>
    <h1>Find Your Next <span>Favorite Movie</span></h1>
    <p>Tell us what you like and we’ll create movie recommendations personalized for you.</p></section>''')
    genre_picker(genres)


def rating_control(movie_id, prefix, label):
    key = f"{prefix}_{movie_id}"
    current = (SKIPPED if movie_id in st.session_state.skipped_movies
               else st.session_state.user_ratings.get(movie_id))
    if key not in st.session_state:
        st.session_state[key] = current
    st.selectbox(label, RATING_OPTIONS, index=None, key=key, format_func=rating_label,
                 placeholder="Choose a rating",
                 on_change=save_rating, args=(movie_id, key))


def rate_movies(movies, labels, movie_ids):
    count = len(st.session_state.user_ratings)
    remaining = RATING_TARGET - count
    genre_mode = st.session_state.discovery_mode == "genres"
    st.html(f'<p class="eyebrow">STEP {2 if genre_mode else 1} OF {3 if genre_mode else 2}</p>')
    st.title("Rate Movies You’ve Seen")
    if genre_mode:
        st.write("Add ratings for movies you’ve watched, or continue with your favorite genres.")
    else:
        st.write("Search for five movies you’ve watched and give each a rating. Half stars are welcome.")
        progress = (f"{count} of {RATING_TARGET} movies rated" if remaining > 0
                    else f"{count} movies rated · goal reached")
        st.progress(min(count / RATING_TARGET, 1.0), text=progress)
    library, ratings_panel = st.columns([2.1, 1], gap="large")
    with library:
        with st.container(border=True, key="rating_library"):
            movie_id = st.selectbox("Find a movie", movie_ids, index=None, key="movie_picker", persist_state="session",
                                    format_func=labels.__getitem__, filter_mode="contains",
                                    placeholder="Type a movie title or year")
            if movie_id is not None:
                if movie_id not in movies:
                    st.warning("This movie is unavailable. Choose another title.")
                else:
                    title, genres = movies[movie_id]
                    st.html(f'<div class="search-preview">{cover_html(movie_id, title)}'
                            f'<div><strong>{escape(labels[movie_id])}</strong>'
                            f'<p>{escape(genre_label(genres))}</p></div></div>')
                    if movie_id in st.session_state.user_ratings:
                        st.caption("Already rated. You can change it in Your Ratings.")
                    else:
                        rating_control(movie_id, "rate", "Your rating")
                        if movie_id in st.session_state.skipped_movies:
                            st.caption("Skipped. Search for another movie to keep going.")
    with ratings_panel:
        with st.container(border=True, key="your_ratings"):
            st.subheader("Your Ratings")
            if not count:
                st.caption("Your ratings will appear here.")
            elif genre_mode:
                st.caption(f"{count} movie{'s' if count != 1 else ''} rated")
            for movie_id in list(st.session_state.user_ratings):
                if movie_id not in movies:
                    st.warning(f"Movie {movie_id} is unavailable.")
                else:
                    _, genres = movies[movie_id]
                    st.text(labels[movie_id])
                    st.caption(genre_label(genres))
                    rating_control(movie_id, "edit", f"Change rating for {labels[movie_id]}")
                st.button("Remove rating", key=f"remove_{movie_id}", on_click=remove_rating,
                          args=(movie_id,))
                st.divider()
    st.divider()
    back, note, submit = st.columns([1, 2.4, 1.6])
    back.button("←  Back", key="back", on_click=go_to, args=(1 if genre_mode else 0,))
    genres_available = bool(selected_genres())
    ready = genres_available if genre_mode else remaining <= 0
    with note:
        if genre_mode and not genres_available:
            st.caption("Go back and choose at least one favorite genre.")
        elif not genre_mode and remaining > 0:
            st.caption(f"Rate {remaining} more movie{'s' if remaining != 1 else ''} to continue.")
    submit.button("Get My Recommendations  →", type="primary", width="stretch", disabled=not ready,
                  key="get_recommendations", on_click=go_to, args=(3,))


def recommendations():
    from src.hybrid_recommender import PreferenceError
    st.html('<p class="eyebrow">PERSONALIZED FOR YOU</p>')
    title, preferences_button, ratings_button = st.columns([2.4, 1, 1])
    title.title("Your Recommendations")
    mode = st.session_state.discovery_mode
    preferences_button.button("Change discovery method", key="update_preferences", width="stretch",
                              on_click=go_to, args=(0,))
    ratings_button.button("Rate More Movies", type="primary", key="rate_more", width="stretch", on_click=go_to, args=(2,))
    genres = selected_genres()
    ratings = st.session_state.user_ratings
    if mode == "movies":
        st.write("These picks are based on the movies you rated. Each card explains the match.")
    elif any(rating != 3 for rating in ratings.values()):
        st.write("These picks use your favorite genres and give more weight to the movies you rated.")
    else:
        st.write("These picks are based on the genres you selected.")
    profile = mode, genres, tuple(sorted(ratings.items()))
    try:
        previous = st.session_state.get("recommendation_result")
        if previous is None or previous[0] != profile:
            use_embeddings = any(rating != 3 for rating in ratings.values())
            results = load_recommender(use_embeddings).recommend_for_new_user(genres, ratings, top_n=6)
            st.session_state.recommendation_result = (profile, results)
        else:
            results = previous[1]
    except PreferenceError as error:
        st.info(str(error))
        return
    except (OSError, ValueError, ImportError, KeyError):
        st.error("We couldn’t prepare recommendations. Check that the MovieLens files and saved model are available.")
        return
    if results.empty:
        st.info("No unrated matches remain. Try updating your preferences.")
        return
    cards = []
    for rank, movie in enumerate(results.itertuples(), 1):
        title_text, year = title_parts(movie.title)
        cards.append(f'''<article class="movie-card">
            {cover_html(movie.movieId, movie.title, rank)}
            <div class="card-content"><h3>{escape(title_text)}</h3>
            <p class="metadata">{escape(year)} · {escape(genre_label(movie.genres))}</p>
            <p class="explanation"><strong>Why this movie?</strong> {escape(movie.explanation)}</p>
            </div></article>''')
    st.html('<div class="movie-grid">' + ''.join(cards) + '</div>')


def main():
    st.set_page_config(page_title="Personalized Movie Recommender", page_icon="🎬", layout="wide")
    st.html(f"<style>{(ROOT / 'assets/styles.css').read_text()}</style>")
    if "discovery_mode" not in st.session_state:
        st.session_state.discovery_mode = None
        st.session_state.step = 0
    for key, initial in [("favorite_genres", []), ("user_ratings", {}),
                         ("skipped_movies", set())]:
        if key not in st.session_state:
            st.session_state[key] = initial
    step = st.session_state.step
    header(step, st.session_state.discovery_mode)
    if step == 0:
        discovery()
    elif step in (1, 2):
        try:
            movies, genres, labels, movie_ids = load_catalog()
        except (OSError, ValueError, KeyError):
            st.error("We couldn’t open the movie catalog. Check that the processed MovieLens catalog is installed.")
            return
        if st.session_state.step == 1:
            preferences(genres)
        else:
            rate_movies(movies, labels, movie_ids)
    else:
        recommendations()
    with st.expander("Credits"):
        logo = ROOT / "assets/tmdb-logo.svg"
        if logo.exists():
            st.image(str(logo), width=95)
        else:
            st.html('<a href="https://www.themoviedb.org"><img alt="TMDB" width="95" src="https://www.themoviedb.org/assets/2/v4/logos/v2/blue_short-8e7b30f73a4020692ccca9c88bafe5dcb6f8a62a4c6bc55cd9ba82bb2cd95f6c.svg"></a>')
        st.caption("Poster data and images: TMDB. This product uses the TMDB API but is not endorsed or certified by TMDB.")
        st.link_button("The Movie Database", "https://www.themoviedb.org")
    if not any(poster_credentials()):
        with st.expander("Enable movie posters"):
            st.write("Add your TMDB API key to .streamlit/secrets.toml as TMDB_API_KEY, or set the TMDB_API_KEY environment variable. The app works without it.")
            st.link_button("TMDB API settings", "https://www.themoviedb.org/settings/api")


if __name__ == "__main__":
    main()
