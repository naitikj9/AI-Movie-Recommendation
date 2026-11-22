# Personalized Movie Recommender

A graduate class project with a working Streamlit app for **new users**:
**Preferences → Rate Movies → Recommendations**. Choose genres, search real MovieLens movies,
rate five watched movies in half-star steps, then explore up to six personalized recommendations.
Genres, ratings, and skips persist while moving between screens in the same session.

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

## Usability changes

| Finding | Cause and fix | Verification |
| --- | --- | --- |
| Genre taps waited for visible feedback | The button style changed after a full script rerun, and a cold session loaded TensorFlow before displaying Preferences. Native genre pills show a selected state on tap, and a Streamlit fragment refreshes the count. A lightweight cached catalog serves onboarding. | Widget tests verify genre selection survives navigation and onboarding never calls the recommender loader. |
| Finding a watched movie took extra steps | Search was a server-side text field with a capped result list and several rating controls. The picker filters the full MovieLens title list as you type. Identical titles have distinct IDs in their labels. | Widget tests select titles from the full catalog and cover editing and removal. |
| No way to mark an unwatched movie | The rating menu only offered stars and an empty state. **Haven’t Seen** is stored separately, so a skip contributes neither to the five-rating target nor to either scoring profile. A skipped movie can still be recommended because it is unwatched. | Tests cover skipping, revisiting a skipped title, replacing a skip with a rating, and changing a rating to a skip. |
| Results were hard to interpret | Cards used generic movie-profile language. Each card now explains selected or inferred genre matches, a named highly rated movie with similar MovieLens rating patterns when supported, or a weaker match. Results are reused within the session for the same inputs. | App tests exercise genre-only and ratings-only results, missing posters, neutral ratings, and invalid IDs. |
| Desktop layout crowded narrow screens | Fixed genre rows and side-by-side controls reduced usable space. Genre pills wrap, rating controls use larger targets, and cards flow into one, two, or three columns. | See the browser checks below. |

With at least one selected genre, visitors can request recommendations before reaching five
ratings. For a ratings-only request, five numeric ratings unlock recommendations. Five neutral
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

1. Open a fresh browser session. On iPhone Safari, tap several genres quickly, deselect one,
   and check that the selected state and count agree. Continue, then go back to verify persistence.
2. With no genres selected, search for watched movies by partial title, try a title that has no
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
