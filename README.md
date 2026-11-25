# 🎬 Bollywood Movie Recommender

**Aaj kya dekhein?** This app helps answer that. Tell it what you like, and it suggests
Bollywood movies for you, or for you and your friends, and shows why it picked them.

> 🙌 This is my version of [AI-Movie-Recommendation](https://github.com/bpisciotti/AI-Movie-Recommendation)
> by bpisciotti, used with permission. I turned it into a Bollywood app and added new features.
> [Who made what](#-credits) is listed at the bottom.

## ✨ What can it do?

| Feature | What you get |
| --- | --- |
| 🎯 **Your picks** | Rate 5 movies you've watched, or just choose your favorite genres |
| 👯 **Group picks** | 2 or 3 friends, one list for everyone |
| 🤔 **Why this movie?** | An explanation and score breakdown on your personal picks |
| 😄 **Mood picker** | Romantic? Full masala? Spooky? Pick a mood and the genres fill in |
| 📼 **Era filter** | From before 1990 to 2020 onwards |
| ❤️ **Watchlist** | Save movies and download the list as CSV |
| 🖼️ **Posters** | Optional, with your own TMDB key |

## 🛠️ Built with

Python · Streamlit · Pandas · NumPy · scikit-learn · Docker · Wikidata · TMDB

## 🚀 Run it

### The easy way: Docker

```bash
docker compose up --build
```

Then open http://localhost:8501. That's it. You don't need an API key for recommendations.

### Without Docker

You need Python 3.12. First create a virtual environment:

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\Activate.ps1` in Windows PowerShell, or
`source .venv/bin/activate` on macOS and Linux). Then:

```bash
python -m pip install -r requirements-docker.txt
python -m streamlit run app.py
```

The file is called `requirements-docker.txt`, but it works here too: it lists what the app needs.

### Want posters? 🖼️

Get your own key from [TMDB API settings](https://www.themoviedb.org/settings/api).
Fill in **one** of `TMDB_API_KEY` or `TMDB_READ_ACCESS_TOKEN` (if you set both, the token is used).

- **Docker:** copy `.env.example` to `.env`, add your key, and run `docker compose up -d`.
- **Without Docker:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`,
  add your key, and restart the app.

Git ignores both files, so keep your key there and never commit it. To find a poster, the app
sends TMDB the movie's ID, not your genres or ratings. Posters don't change the ranking. If a
poster can't be loaded, you'll see a placeholder.

## 🧠 How does it pick movies?

In short: **movies with genres you like, with a little help from popularity.** It's a
content-based recommender and the app doesn't use a trained model.

1. **It learns your taste.** A movie you rate above 3 pulls the list toward its genres. Below 3
   pushes away. Exactly 3 doesn't move it.
2. **It checks the genre match.** It uses cosine similarity between your taste and each movie's
   genres, scaled to 0–1 with `(cosine + 1) / 2`.
3. **It adds your favorite genres.** If you only choose genres, that match is your preference
   score. If your ratings count too, it's 75% ratings and 25% favorite genres.
4. **It breaks the crowd.** Lots of movies have the same genres, so the final score is
   **70% preference + 30% popularity**.

Popularity means the number of Wikipedia and sister-site pages about a movie (log-scaled). It
shows how widely known a movie is, not how good it is. Movies you've rated won't be suggested
again. The score is only for ranking: it isn't a star rating or an accuracy percentage.

### 👯 What about groups?

Each person adds favorite genres, rated movies, or both. Everyone's scores are converted to
percentiles so they can be compared fairly. Then you choose:

- **Best for the group on average:** the highest average wins.
- **Keep everyone happy:** the movie's lowest match decides its rank ("least misery").

If anyone in the group has rated a movie, it's left out.

## 🍿 Where do the movies come from?

The app has **3,415 films** from [Wikidata](https://www.wikidata.org) (CC0), stored in
`data/bollywood/movies.csv`. To fetch a fresh copy:

```bash
python scripts/fetch_bollywood.py
```

The count can change when Wikidata changes. I use "Hindi-language films from India" to mean
Bollywood, which is close but not exact. Films without a usable genre or date are left out.

## 📁 What's where?

```text
app.py                      The Streamlit app
src/hybrid_recommender.py   Scoring and explanations
src/group_recommender.py    Group picks
src/discovery.py            Moods, era filter, watchlist download
src/posters.py              TMDB posters
scripts/fetch_bollywood.py  Builds the movie catalog
data/bollywood/             The movies
tests/                      Tests
```

`notebooks/`, `models/`, `data/raw/` and `data/processed/` are from the original MovieLens
project. The running app doesn't use them, and they aren't in the Docker image. MovieLens has
its own [usage conditions](data/raw/README.txt).

## 🧪 Tests

Some tests check the original saved model, so you also need TensorFlow and Keras:

```bash
python -m pip install -r requirements-docker.txt tensorflow==2.21.0 keras==3.15.1
python -m unittest discover -s tests -v
```

## 🚧 What it can't do yet

- I haven't measured how good the recommendations are with an evaluation study.
- It only knows genres and popularity for ranking, not cast, director or story.
- Famous movies get an advantage because of the popularity score.
- There are no accounts, so refreshing the page may clear your ratings and watchlist.
- Posters need a TMDB key and a connection to TMDB.

## 🤝 Credits

| From the [original project](https://github.com/bpisciotti/AI-Movie-Recommendation) by bpisciotti | Added in my version |
| --- | --- |
| Streamlit app and interface | Bollywood catalog from Wikidata |
| New-user recommendation flows and scoring | Popularity in the ranking |
| Optional poster lookup | Group picks |
| Notebooks and trained MovieLens model | Score breakdown |
| Original tests | Mood picker, era filter and watchlist |
| | Docker setup without TensorFlow |
| | Tests for the new features |

The original repository had no license file when I made this version. The permission mentioned
above was for this version, so please ask the original author before reusing their code.

This product uses the TMDB API but is not endorsed or certified by TMDB.
