# Personalized Movie Recommender

A graduate class project with a working Streamlit app for **new users**:
**Preferences → Rate Movies → Recommendations**. Choose genres, search real MovieLens movies,
add half-star ratings, then explore six personalized recommendations. Genres and ratings persist
while moving between screens in the same session. They are not saved as an account or to disk.

The interface follows the supplied Figma prototype screenshots: near-black backgrounds, coral
accents, genre tiles, a movie library beside Your Ratings, and a three-column recommendation grid.
Optional TMDB posters appear when a credential is configured. Decorative placeholders are used
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

Model files are ignored by Git in this project. Copy the saved artifacts separately when moving
the app to another machine. Do not train a replacement just to launch the UI. Startup checks
mapping dimensions and gives a setup message if files are missing. The app never calls `fit()`
or saves, overwrites, or modifies model weights.

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

A new person has no trained user embedding. We never invent a MovieLens user ID or borrow
another person's embedding. Instead we combine two request-local profiles:

1. **Genre profile.** Each selected genre contributes 1 to that genre's coordinate. A rated movie
   contributes its unit-length genre vector multiplied by `rating - 3.0`. Sum rated contributions
   and divide by their total absolute weight, then add the explicit genre selections. This keeps
   a very long rating history from overwhelming explicitly selected genres.
2. **Movie-space profile.** Look up each rated movie using the saved mapping. Sum its unit-length
   learned movie embedding multiplied by `rating - 3.0`, then divide by total absolute weight.
   Missing embeddings are skipped, not assigned invented factors.

We deliberately use **3 stars as the interaction's neutral point**, not the historical dataset
mean (~3.5). It makes the UI convention simple: above 3 means positive, below 3 means negative,
and exactly 3 contributes zero. A 0.5 rating has weight -2.5; a 5 has weight +2. Explicit genres
can reinforce or counter the genres inferred from ratings. This is a documented heuristic.

For each candidate, compute cosine similarity to each available profile. Convert signed cosine
from `[-1,1]` to `[0,1]` using `(cosine + 1) / 2` (zero similarity becomes 0.5). Combine the scores:

```text
hybrid_score = 0.50 * content_score + 0.50 * embedding_score
```

Weights are configurable, nonnegative, and normalized to sum to 1. Missing or zero-norm signals
are removed from the denominator and remaining weights are renormalized **per candidate**.
Genre-only requests therefore use content only. A metadata movie without embeddings can still
receive a content score; a movie without genres can still receive an embedding score. Candidates
without any enabled signal are excluded. A zero-information profile (e.g., no genres and only
neutral ratings) raises a friendly `PreferenceError`. If weights explicitly disable the only
available signal, the request also returns a clear error.

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

Returned DataFrames include IDs, titles, genres, content/embedding/hybrid scores, explanations,
rating counts, and average historical ratings. Missing component scores are `NaN`, not invented
values. Rated movies are always excluded. Ranking sorts by hybrid score, then historical rating
count only for exact ties, then movieId. Popularity never adds to the personalization score.

**Hybrid scores are ranking scores, not stars, probabilities, or accuracy percentages.** The UI
hides these technical columns. Explanations come directly from positive genre-profile matches
and/or signed movie-space similarities; they are deterministic and do not use an LLM. The learned
movie-space profile is a **cold-start approximation**, not a new learned user embedding or a rating
prediction from the original network.

## App code: where to look

- `src/hybrid_recommender.py`: artifact loading, ID validation, profiles, scoring, explanations,
  search, and known-user inference. No Streamlit dependency or mutable per-user state.
- `app.py`: `load_recommender()` caches the read-only engine with `st.cache_resource`.
  `preferences()`, `rate_movies()`, and `recommendations()` render the three screens.
  Callbacks store persistent selections separately from ephemeral widget keys in session state.
- `assets/styles.css` and `.streamlit/config.toml`: prototype-inspired layout and dark theme.
- `tests/`: synthetic scoring tests and real-app interaction tests.

The app encourages five ratings. It also permits an earlier genre-based fallback if at least one
genre is selected; adding ratings improves the available personalization signals. Ratings-only
users are asked for five ratings. Unknown/no-result searches and uninformative profiles have clear
messages. Duplicate titles show a movie ID in the rating screen to distinguish records. Search
matches literal title or genre text. Results are capped at 12; narrow the query to find a movie.

## Verification

```bash
.venv.nosync/bin/python -m unittest discover -s tests -v
```

Tests cover genre-only/rating-only/hybrid requests, dislikes, invalid IDs/ratings/weights,
self-exclusion, missing embeddings/genres, duplicate titles, ranking, result count, and empty
preferences. UI tests exercise the real saved artifacts without training them. The browser flow
was also checked visually. Model/data/notebook SHA-256 hashes are checked during integration
verification to ensure protected files remain unchanged.

## Limitations

The hybrid has not undergone independent ranking-quality evaluation. Prior collaborative MAE/RMSE
results describe known-user rating prediction; they do **not** establish this cold-start hybrid's
quality. Movie-space cosine uses a heuristic geometry and leaves out trained user factors and
biases. Sparse histories and broad genres limit precision, and negative/cancelling profiles can
be uninformative. A single negative rating can produce less-similar suggestions, but that is not
a substitute for positive preferences. Historical rating counts are biased toward popular movies.

Preferences live only for a browser session; a refresh/reconnect may reset them. Account storage, deployment, and a hybrid evaluation study remain future work.
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
