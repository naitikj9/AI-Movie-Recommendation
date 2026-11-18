"""Streamlit entry point. Run with: python -m streamlit run app.py."""
from html import escape
from pathlib import Path
import re
import os

import streamlit as st

from src.hybrid_recommender import HybridRecommender, PreferenceError
from src.posters import fetch_poster, load_tmdb_mapping

ROOT = Path(__file__).resolve().parent
RATING_OPTIONS = [None] + [value / 2 for value in range(1, 11)]


@st.cache_resource(show_spinner="Opening your movie library…")
def load_recommender():
    return HybridRecommender.from_project(ROOT)


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
    return load_tmdb_mapping(ROOT / "data/raw/links.csv")


@st.cache_data(ttl=3600, max_entries=1000, show_spinner=False)
def cached_poster(tmdb_id, api_key, read_token):
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
    mini = rank is None
    css_class = "mini-cover" if mini else "movie-cover"
    rank_badge = f'<span class="rank">{rank:02d}</span>' if rank is not None else ""
    if poster:
        return (f'<div class="{css_class} has-poster">'
                f'<img src="{poster}" alt="{escape(title, quote=True)} poster" />'
                f'{rank_badge}</div>')
    decoration = "▶" if mini else '<span class="cover-mark">▶</span><span class="cover-label">MOVIE NIGHT</span>'
    return (f'<div class="{css_class} palette-{int(movie_id) % 6}" aria-label="Poster unavailable">'
            f'{rank_badge}{decoration}</div>')


def go_to(step):
    st.session_state.step = step


def toggle_genre(genre):
    selected = set(st.session_state.favorite_genres)
    selected.symmetric_difference_update({genre})
    st.session_state.favorite_genres = sorted(selected)


def save_rating(movie_id, widget_key):
    value = st.session_state[widget_key]
    if value is None:
        st.session_state.user_ratings.pop(movie_id, None)
    else:
        st.session_state.user_ratings[movie_id] = float(value)


def save_search():
    st.session_state.search_query = st.session_state.movie_search


def remove_rating(movie_id):
    st.session_state.user_ratings.pop(movie_id, None)
    for prefix in ["rate", "edit"]:
        st.session_state.pop(f"{prefix}_{movie_id}", None)


def title_parts(title):
    match = re.search(r"\s*\((\d{4})\)\s*$", title)
    return (title[:match.start()], match.group(1)) if match else (title, "Year unavailable")


def genre_label(genres):
    return "Genres unavailable" if genres == "(no genres listed)" else genres.replace("|", " · ")


def rating_label(value):
    if value is None:
        return "Choose a rating"
    return f"{'★' * int(value)}{'½' if value % 1 else ''}  {value:g} / 5"


def header(step):
    st.html('''<header class="brand-bar"><div class="brand"><span class="brand-icon">▶</span>
    <div><strong>PERSONALIZED</strong><span>Movie Recommender</span></div></div>
    <span class="brand-tag">Your next favorite is waiting</span></header>''')
    items = []
    for index, label in enumerate(["Preferences", "Rate Movies", "Recommendations"], 1):
        state = "complete" if index < step else "active" if index == step else "future"
        items.append(f'<div class="step {state}"><b>{"✓" if index < step else index}</b><span>{label}</span></div>')
    st.html('<nav aria-label="Recommendation progress" class="stepper">' + '<i></i>'.join(items) + '</nav>')


def preferences(recommender):
    st.html('''<section class="hero"><p class="eyebrow">CURATED FOR YOU</p>
    <h1>Find Your Next <span>Favorite Movie</span></h1>
    <p>Tell us what you like and we’ll create movie recommendations personalized for you.</p></section>''')
    with st.container(border=True, key="genre_panel"):
        title, count = st.columns([4, 1])
        with title:
            st.subheader("What do you love to watch?")
            st.caption("Select as many as you like. You can update these later.")
        with count:
            st.html(f'<span class="selection-count">{len(st.session_state.favorite_genres)} selected</span>')
        genres = recommender.genres
        for start in range(0, len(genres), 4):
            for column, genre in zip(st.columns(4), genres[start:start + 4]):
                selected = genre in st.session_state.favorite_genres
                column.button(f"{'✓' if selected else '＋'}  {genre}", key=f"genre_{genre}",
                              type="primary" if selected else "secondary", width="stretch",
                              on_click=toggle_genre, args=(genre,))
    _, next_column = st.columns([4, 1])
    next_column.button("Continue  →", type="primary", width="stretch", key="continue",
                       on_click=go_to, args=(2,))
    if not st.session_state.favorite_genres:
        st.caption("You can also start with movie ratings instead of genres.")


def rating_control(movie_id, prefix, value=None):
    key = f"{prefix}_{movie_id}"
    if key not in st.session_state:
        st.session_state[key] = value
    st.selectbox("Your rating", RATING_OPTIONS, key=key, format_func=rating_label,
                 on_change=save_rating, args=(movie_id, key))


def rate_movies(recommender):
    heading, progress = st.columns([4, 1])
    count = len(st.session_state.user_ratings)
    with heading:
        st.html('<p class="eyebrow">STEP 2 OF 3</p>')
        st.title("Rate Movies You’ve Seen")
        st.write("Rate a few movies you’ve watched so we can better understand your preferences.")
        st.caption("Aim for at least 5 movies. Half-star ratings are welcome.")
    with progress:
        with st.container(border=True):
            st.subheader(f"{count} of 5")
            st.caption("movies rated")
            st.progress(min(count / 5, 1.0))
    library, ratings_panel = st.columns([2.1, 1], gap="large")
    with library:
        with st.container(border=True, key="rating_library"):
            if "movie_search" not in st.session_state:
                st.session_state.movie_search = st.session_state.search_query
            query = st.text_input("Find a movie", placeholder="Search by movie title or genre",
                                  key="movie_search", on_change=save_search)
            st.markdown("#### Search results" if query.strip() else "#### Suggested movies")
            search = recommender.search_movies(query, limit=12 if query.strip() else 6)
            if search.empty:
                st.info("No movies found. Try a shorter title or a genre.")
            for movie in search.itertuples():
                title, year = title_parts(movie.title)
                details, control = st.columns([1.6, 1])
                with details:
                    st.html(f'<div class="library-movie">{cover_html(movie.movieId, movie.title)}'
                            f'<div><h4>{escape(title)}</h4><p>{escape(year)} · {escape(genre_label(movie.genres))}</p>'
                            f'<small>Movie ID {movie.movieId}</small></div></div>')
                with control:
                    # Rated movies are edited in one place to avoid conflicting widget values.
                    if movie.movieId in st.session_state.user_ratings:
                        st.caption(f"Rated {st.session_state.user_ratings[movie.movieId]:g} ★ — edit in Your Ratings")
                    else:
                        rating_control(movie.movieId, "rate")
                st.divider()
    with ratings_panel:
        with st.container(border=True, key="your_ratings"):
            st.subheader("Your Ratings")
            st.caption(f"{count} total")
            if not count:
                st.caption("Your rated movies will appear here. Choose a rating to add one.")
            movie_lookup = recommender.movies.set_index("movieId")
            for movie_id, rating in list(st.session_state.user_ratings.items()):
                movie = movie_lookup.loc[movie_id]
                title, year = title_parts(movie.title)
                st.markdown(f"**{title}**")
                st.caption(f"{year} · {genre_label(movie.genres)} · Movie ID {movie_id}")
                rating_control(movie_id, "edit", rating)
                st.button("Remove", key=f"remove_{movie_id}", on_click=remove_rating, args=(movie_id,))
                st.divider()
    st.divider()
    back, note, submit = st.columns([1, 2.4, 1.6])
    back.button("←  Back", key="back", on_click=go_to, args=(1,))
    genres_available = bool(st.session_state.favorite_genres)
    ready = count >= 5 or genres_available
    with note:
        if count < 5 and genres_available:
            st.caption("You can continue with genres. More ratings make your recommendations more personal.")
        elif count < 5:
            st.caption(f"Rate {5 - count} more movie{'s' if 5-count != 1 else ''}, or go back and choose genres.")
    submit.button("Get My Recommendations  →", type="primary", width="stretch", disabled=not ready,
                  key="get_recommendations", on_click=go_to, args=(3,))


def recommendations(recommender):
    st.html('<p class="eyebrow">PERSONALIZED FOR YOU</p>')
    title, preferences_button, ratings_button = st.columns([2.4, 1, 1])
    title.title("Your Recommendations")
    preferences_button.button("Update Preferences", key="update_preferences", width="stretch", on_click=go_to, args=(1,))
    ratings_button.button("Rate More Movies", type="primary", key="rate_more", width="stretch", on_click=go_to, args=(2,))
    count = len(st.session_state.user_ratings)
    rated_context = f" and the {count} movie{'s' if count != 1 else ''} you rated" if count else ""
    st.write(f"Movies selected based on your preferences{rated_context}.")
    try:
        results = recommender.recommend_for_new_user(
            st.session_state.favorite_genres, st.session_state.user_ratings, top_n=6
        )
    except PreferenceError as error:
        st.info(str(error))
        return
    if results.empty:
        st.info("No unrated matches remain. Try updating your preferences.")
        return
    for start in range(0, len(results), 3):
        for column, (_, movie) in zip(st.columns(3, gap="large"), results.iloc[start:start + 3].iterrows()):
            title_text, year = title_parts(movie.title)
            rank = int(movie.name) + 1
            with column:
                st.html(f'''<article class="movie-card">
                {cover_html(movie.movieId, movie.title, rank)}
                <div class="card-content"><h3>{escape(title_text)}</h3>
                <p class="metadata">{escape(year)} · {escape(genre_label(movie.genres))}</p>
                <div class="explanation"><span>ⓘ</span> {escape(movie.explanation)}</div></div></article>''')
    st.caption("Movie posters appear when available; decorative artwork is used otherwise. Recommendations are suggestions based on your taste.")


def main():
    st.set_page_config(page_title="Personalized Movie Recommender", page_icon="🎬", layout="wide")
    st.html(f"<style>{(ROOT / 'assets/styles.css').read_text()}</style>")
    for key, initial in [("step", 1), ("favorite_genres", []), ("user_ratings", {}), ("search_query", "")]:
        if key not in st.session_state:
            st.session_state[key] = initial
    header(st.session_state.step)
    try:
        recommender = load_recommender()
    except (OSError, ValueError, ImportError, KeyError) as error:
        st.error("We couldn’t open the movie library. Check that the saved model, mappings, and MovieLens files are installed.")
        with st.expander("Setup details"):
            st.code(str(error))
        st.stop()
    if st.session_state.step == 1:
        preferences(recommender)
    elif st.session_state.step == 2:
        rate_movies(recommender)
    else:
        recommendations(recommender)
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
