# Personalized Movie Recommender

A graduate class project with a working Streamlit app for **new users**:
**Choose a discovery method → Set preferences → Explore recommendations**. Start with five
watched-movie ratings, or pick favorite genres and add ratings optionally. Both paths offer
up to six recommendations. Switching paths keeps ratings and clears explicit genres.

The interface follows the supplied Figma prototype screenshots: near-black backgrounds, coral
accents, wrapping genre toggles, a movie picker beside Your Ratings on desktop, and a
responsive recommendation grid.
Optional TMDB posters appear in the selected search result preview and in recommendations when
a credential is configured. Decorative placeholders are used
when posters are unavailable. No API key, Figma connection, or notebook execution is required
for the recommender itself.

## Run locally

From this project's root, use the existing Python 3.12 environment:

```bash
.venv.nosync/bin/python -m pip install -r requirements.txt
.venv.nosync/bin/python -m streamlit run app.py
```

Open the local address printed by Streamlit (normally http://localhost:8501).
If dependencies are already installed, only the second command is needed. The `.nosync` name
keeps the environment out of iCloud syncing on this Mac. For another computer, create a Python
3.12 virtual environment and install the same requirements there. TensorFlow needs a compatible
platform; use the existing working versions rather than upgrading packages unnecessarily.
`requirements.txt` already includes Streamlit, TensorFlow/Keras, Pandas, NumPy and scikit-learn.

### Required existing artifacts

```text
data/processed/movies_processed.csv   # movieId, title, genres
data/raw/ratings.csv                  # userId, movieId, rating (timestamp may also exist)
models/movie_recommender.keras        # selected trained Experiment 2
models/movie_to_index.json            # exact movieId → embedding row
models/user_to_index.json             # exact userId → embedding row
```

Keep the saved artifacts and CSVs available in the deployed branch. The Preferences and Rate
Movies screens use the catalog independently of the model. When collaborative recommendations
first need the model, the loader checks the mappings. The app opens the model for read-only use.

## How the models work

### Content-based baseline (notebook 02)

The `genres` string is split at `|`. `MultiLabelBinarizer` creates one binary column per real genre;
`(no genres listed)` becomes an all-zero row. Cosine similarity finds movies with similar genre
sets. The module reuses that preprocessing method, rather than relying on notebook variables.
`movieId` identifies movies throughout; titles can be duplicated.

### Trained collaborative model (notebook 03)

The saved model learns user/movie embeddings from historical ratings. It adds their dot product,
user/movie biases, and the training mean to predict a rating. Selected Experiment 2 uses 16 factors
and Adam learning rate 0.005. Known users can still use the original inference route:

```python
from src.hybrid_recommender import HybridRecommender

engine = HybridRecommender.from_project(".")
known_user_movies = engine.recommend_for_known_user(user_id=1, top_n=10)
```

This method uses the original mappings and network and clips **predicted ratings** to 0.5–5.0.
It excludes that user's historically rated movies. The public UI focuses on new users instead.

### New-user hybrid (src/hybrid_recommender.py)

A new visitor’s profile is built from the movies they rate and, in the favorite-genres
path, the genres they select. It uses learned movie embeddings from the saved model.
There is no user-embedding lookup for this flow.

1. **Ratings.** Each rated movie contributes its unit-length genre vector and learned movie
   embedding, scaled by its rating minus 3. Ratings above 3 pull toward similar titles, ratings
   below 3 push away, and a 3 adds no directional signal. Compute cosine similarity to the
   two profiles and blend the available genre and embedding scores 50/50 by default.
2. **Favorite genres.** Compute a separate cosine similarity to a vector with equal entries for
   the visitor’s chosen genres. Adding genres changes the profile’s direction without increasing
   its share of the final score.

Each cosine is mapped to `[0, 1]` with `(cosine + 1) / 2`. Zero similarity maps to 0.5.
The watched-movies path uses the ratings score alone, including any genre patterns inferred
from rated movies. The favorite-genres path uses the genre score alone when there is no usable
rating signal. When both signals are available, the formula is:

```text
hybrid_score = 0.75 * ratings_score + 0.25 * favorite_genre_score
```

The previous formula added 1 for each explicit genre to a ratings-derived genre vector whose
norm was bounded by 1. The explicit vector therefore had norm `sqrt(number of genres)`.
The 50/50 genre-versus-embedding blend did not give genres and ratings equal influence.
The separate score removes this effect. The cosine-to-score conversion applies the same
linear transformation to both signals.

Within the ratings score, an unavailable genre or embedding component is omitted and the
remaining component receives its weight. A candidate without genre metadata can use its
ratings score alone. If only its favorite-genre score is available in a mixed request,
its ratings contribution is neutral (0.5). Candidates without any enabled signal are
excluded. A zero-information profile raises a friendly `PreferenceError`.

```python
movies = engine.recommend_for_new_user(
    favorite_genres=["Action", "Sci-Fi"],
    user_ratings={1: 4.5, 2: 1.0},  # raw movieId → rating
    top_n=6,
    content_weight=0.5,
    embedding_weight=0.5,
    min_rating_count=0,  # optional reliability filter, off by default
)
```

Returned DataFrames include IDs, titles, genres, content, embedding, favorite-genre and hybrid
scores, explanations, rating counts, and average historical ratings. Missing component scores are `NaN`, not invented
values. Rated movies are always excluded. Ranking sorts by hybrid score, then historical rating
count only for exact ties, then movieId. Popularity never adds to the personalization score.

**Hybrid scores are ranking scores, not stars, probabilities, or accuracy percentages.** The UI
hides these technical columns. Explanations come directly from positive genre-profile matches
and/or signed movie-space similarities; they are deterministic and do not use an LLM. The learned
movie-space profile is a **cold-start approximation**, not a new learned user embedding or a rating
prediction from the original network.

## Usability changes

| Finding | Cause and fix | Verification |
| --- | --- | --- |
| Genre taps waited for visible feedback | The button style changed after a full script rerun, and a cold session loaded TensorFlow before displaying Preferences. Native genre pills show a selected state on tap, and a Streamlit fragment refreshes the count. A lightweight cached catalog serves onboarding. | Widget tests verify genre selection survives navigation and onboarding never calls the recommender loader. |
| Finding a watched movie took extra steps | Search was a server-side text field with a capped result list and several rating controls. The picker filters the full MovieLens title list as you type. Identical titles have distinct IDs in their labels. | Widget tests select titles from the full catalog and cover editing and removal. |
| No way to mark an unwatched movie | The rating menu only offered stars and an empty state. **Haven’t Seen** is stored separately, so a skip contributes neither to the five-rating target nor to either scoring profile. A skipped movie can still be recommended because it is unwatched. | Tests cover skipping, revisiting a skipped title, replacing a skip with a rating, and changing a rating to a skip. |
| Results were hard to interpret | Cards used generic movie-profile language. Each card now explains selected or inferred genre matches, a named highly rated movie with similar MovieLens rating patterns when supported, or a weaker match. Results are reused within the session for the same inputs. | App tests exercise genre-only and ratings-only results, missing posters, neutral ratings, and invalid IDs. |
| Desktop layout crowded narrow screens | Fixed genre rows and side-by-side controls reduced usable space. Genre pills wrap, rating controls use larger targets, and cards flow into one, two, or three columns. | See the browser checks below. |

In the favorite-genres path, at least one selection unlocks optional ratings and recommendations.
In the watched-movies path, five numeric ratings unlock recommendations. Five neutral
ratings give the ranking no preference direction, so the results page asks for a genre or a
rating above or below three stars.

## Verification

Run the suite from the project root.

```bash
.venv.nosync/bin/python -m unittest discover -s tests -v
```

The suite was verified with Python 3.12 and Streamlit 1.65. It exercises the actual saved
Keras artifact, including known-user predictions, and
checks model and mapping hashes around those tests. Poster tests use controlled responses to
cover missing artwork, invalid links, malformed responses, and network failure.

Local Chrome testing covered title search, skipping a movie, and opening genre-based
recommendations. At 390 px, the first genre count update took about 110 ms. Recommendation
cards fit without horizontal scrolling at 320, 390, 820, and 1280 px. The deployed app and
physical iPhone Safari are still to be checked.

Use this checklist after starting locally or pushing the update to the GitHub branch
connected to Community Cloud.

1. Open a fresh browser session and choose the favorite-genres path. On iPhone Safari, tap
   several genres quickly, deselect one, and check that the selected state and count agree.
   Continue, then go back to verify persistence.
2. Switch to the watched-movies path, search by partial title, try a title that has no
   matches, and choose **Haven’t Seen** for one title. Check that the count stays the same and
   that the results button becomes available after five numeric ratings.
3. Repeat with a genre and fewer than five ratings. Check the card explanations and verify that
   rated titles are excluded. On mobile, check for readable controls and no horizontal scrolling.
4. On Community Cloud, open **Manage app** to check logs for catalog or artifact-loading errors
   after testing both the genre-only and ratings-only flows.
   [Streamlit Community Cloud app management](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## Limitations

The hybrid has not undergone independent ranking-quality evaluation. Prior collaborative MAE/RMSE
results describe known-user rating prediction; they do **not** establish this cold-start hybrid's
quality. Movie-space cosine uses a heuristic geometry and leaves out trained user factors and
biases. Sparse histories and broad genres limit precision, and negative/cancelling profiles can
be uninformative. A single negative rating can produce less-similar suggestions, but that is not
a substitute for positive preferences. Historical rating counts are biased toward popular movies.

Preferences live only for a browser session; a refresh/reconnect may reset them. Account storage and a hybrid evaluation study remain future work.
Poster availability depends on optional TMDB credentials and network access. MovieLens has its own
usage conditions; see `data/raw/README.txt`. No notebook or raw-data changes are needed to run.


## Enable real movie posters (optional)

1. Get your own v3 API key from [TMDB API settings](https://www.themoviedb.org/settings/api).
   Registration instructions are in [TMDB's documentation](https://developer.themoviedb.org/docs/getting-started).
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Set the credential locally:

   ```toml
   TMDB_API_KEY = "your_v3_api_key_here"
   ```

   Alternatively set `TMDB_READ_ACCESS_TOKEN` to your API Read Access Token, or set either name
   as an environment variable. Environment values take priority; a read token takes priority over
   the v3 key if both are supplied. Do not paste credentials into chat or commit them. The local
   secrets file is excluded by `.gitignore`.
4. Restart Streamlit and open Rate Movies or Recommendations.

Posters are mapped **movieId → tmdbId** using `data/raw/links.csv`, so duplicate titles are not
confused. `src/posters.py` requests TMDB movie details, fetches and verifies the poster image,
and returns image bytes as a data URI. Credentials never reach browser markup. Requests have
short timeouts, and results are cached in memory for one hour (including unavailable results).
Changing a credential changes the cache key; retry transient failures by clearing Streamlit's
cache or waiting for expiry. Missing credentials/links/images, invalid credentials, rate limits,
and network failures retain the local fallback without stopping recommendations.

Only movie IDs are sent to TMDB—not your genres, ratings, or model files. Posters do not affect
recommendation ranking or modify the neural network. Requests and Pillow are already existing
dependencies. The credits section identifies TMDB and includes its non-endorsement notice.
See [TMDB image documentation](https://developer.themoviedb.org/docs/image-basics).

## Credits and additions

Based on [AI-Movie-Recommendation](https://github.com/bpisciotti/AI-Movie-Recommendation)
by bpisciotti, used with permission. The sections above describe the original project.

### Added in this fork

- **Group picks:** two or three people get one shared list. Each person's scores become percentiles,
  then the list is ranked by the group average or by the lowest individual match (least misery).
  Movies anyone has rated are excluded.
- **Score breakdown:** each "Why this movie?" card shows how its ranking points add up.
- **Docker:** run `docker compose up --build`, then open http://localhost:8501. It installs
  `requirements-docker.txt`.

### Bollywood catalog

The app in this fork recommends **only Hindi-language films from India**. This replaces the
MovieLens behavior described in the earlier sections.

- Data comes from [Wikidata](https://www.wikidata.org) (CC0), saved in `data/bollywood/movies.csv`.
  Rebuild it with `python scripts/fetch_bollywood.py`. "Hindi film from India" is a close proxy
  for Bollywood, not an exact label, and films without a usable genre or date are left out.
- Recommendations are **genre-based**. The catalog has no user ratings, so the trained MovieLens
  model is not used by the app. Its code and tests remain in the repository.
- Score = 70% genre match + 30% popularity (log-scaled count of Wikipedia and sister-site pages),
  so equal genre matches no longer all score 100. Popularity measures reach, not quality.

### Discovery features

- **Posters:** copy `.env.example` to `.env` and add a TMDB credential (Docker), or use
  `.streamlit/secrets.toml` (local runs). Films without a TMDB link keep the placeholder.
- **Release era:** filters recommendations and group picks by year.
- **Mood picker:** pre-selects genres for a mood; you can still change them.
- **Watchlist:** save recommendations and download them as CSV. It lasts for the browser session.
- **Smaller Docker image:** the app does not use the trained model, so the image leaves out
  TensorFlow and `models/`.

