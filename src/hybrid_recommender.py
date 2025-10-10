"""Read-only movie recommendations, independent of notebook or Streamlit state."""
from __future__ import annotations

import json
from numbers import Integral, Real
from pathlib import Path
from threading import RLock
from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


FAVORITE_GENRE_WEIGHT = 0.25


class PreferenceError(ValueError):
    """A request needs different or more useful preferences."""


def _integer_id(value):
    if isinstance(value, bool):
        raise PreferenceError("Movie and user IDs must be integers.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    raise PreferenceError("Movie and user IDs must be integers.")


def _unit_rows(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values, dtype=float), where=norms > 1e-12)


class HybridRecommender:
    """Blend genre and learned-movie-space cosine scores for new users.

    No model training occurs. A temporary movie-space profile is an approximation,
    not a learned user embedding. Methods keep request state local so a Streamlit
    cached instance can be shared safely.
    """

    def __init__(self, movies, movie_to_index, movie_embeddings, *,
                 ratings=None, model=None, user_to_index=None):
        required = {"movieId", "title", "genres"}
        if not required.issubset(movies.columns):
            raise ValueError("Movie metadata must include movieId, title, genres.")
        self._movies = movies[["movieId", "title", "genres"]].copy().reset_index(drop=True)
        if self._movies.movieId.isna().any() or not self._movies.movieId.is_unique:
            raise ValueError("Movie IDs must be present and unique.")
        self._movies["movieId"] = self._movies.movieId.map(_integer_id)
        self._movies["title"] = self._movies.title.fillna("Untitled movie").astype(str)
        self._movies["genres"] = self._movies.genres.fillna("(no genres listed)")
        self._positions = {movie_id: row for row, movie_id in enumerate(self._movies.movieId)}
        self._genre_lists = self._movies.genres.map(
            lambda value: [g for g in str(value).split("|") if g and g != "(no genres listed)"]
        )
        encoder = MultiLabelBinarizer()
        self._genre_features = encoder.fit_transform(self._genre_lists).astype(float)
        self.genres = tuple(encoder.classes_.tolist())
        self._genre_units = _unit_rows(self._genre_features)
        self._genre_columns = {genre: column for column, genre in enumerate(self.genres)}
        self._movie_to_index = {_integer_id(k): int(v) for k, v in movie_to_index.items()}
        embeddings = np.asarray(movie_embeddings, dtype=float)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("Movie embeddings must be a finite matrix.")
        indexes = list(self._movie_to_index.values())
        if sorted(indexes) != list(range(len(embeddings))):
            raise ValueError("Saved movie mapping must match the embedding rows exactly.")
        # Align embeddings to metadata via movieId, never title or CSV row order.
        self._embedding_features = np.zeros((len(self._movies), embeddings.shape[1]))
        self._has_embedding = np.zeros(len(self._movies), dtype=bool)
        for movie_id, row in self._positions.items():
            if movie_id in self._movie_to_index:
                self._embedding_features[row] = embeddings[self._movie_to_index[movie_id]]
                self._has_embedding[row] = np.linalg.norm(self._embedding_features[row]) > 1e-12
        self._embedding_units = _unit_rows(self._embedding_features)
        self._model = model
        self._user_to_index = {int(k): int(v) for k, v in (user_to_index or {}).items()}
        self._model_lock = RLock()
        self._ratings = (ratings[["userId", "movieId", "rating"]].copy()
                         if ratings is not None else pd.DataFrame(columns=["userId", "movieId", "rating"]))
        if not self._ratings.empty:
            statistics = self._ratings.groupby("movieId").rating.agg(
                average_rating="mean", rating_count="count"
            )
            self._movies = self._movies.join(statistics, on="movieId")
        else:
            self._movies["average_rating"] = np.nan
            self._movies["rating_count"] = 0
        self._movies["rating_count"] = self._movies.rating_count.fillna(0).astype(int)
        for array in [self._genre_features, self._genre_units, self._embedding_features,
                      self._embedding_units, self._has_embedding]:
            array.setflags(write=False)

    @classmethod
    def from_project(cls, root, *, use_embeddings=True):
        """Load saved artifacts once; the caller can cache this resource."""
        root = Path(root)
        movies = pd.read_csv(root / "data/processed/movies_processed.csv")
        ratings = pd.read_csv(root / "data/raw/ratings.csv", usecols=["userId", "movieId", "rating"])
        if not use_embeddings:
            return cls(movies, {}, np.empty((0, 0)), ratings=ratings)

        from tensorflow import keras

        movie_mapping = json.loads((root / "models/movie_to_index.json").read_text())
        user_mapping = json.loads((root / "models/user_to_index.json").read_text())
        model = keras.models.load_model(root / "models/movie_recommender.keras", compile=False)
        embedding_weights = model.get_layer("movie_embedding").get_weights()[0]
        if len(user_mapping) != model.get_layer("user_embedding").input_dim:
            raise ValueError("Saved user mapping does not match the model.")
        if sorted(int(v) for v in user_mapping.values()) != list(range(len(user_mapping))):
            raise ValueError("User indexes must match the trained embedding rows.")
        return cls(movies, movie_mapping, embedding_weights, ratings=ratings,
                   model=model, user_to_index=user_mapping)

    @property
    def movies(self):
        """Return a copy: callers cannot alter the shared cached metadata."""
        return self._movies.copy()

    def search_movies(self, query="", limit=6):
        """Literal, case-insensitive title/genre search; duplicate titles stay distinct."""
        query = str(query).strip()
        results = self._movies
        if query:
            mask = (results.title.str.contains(query, case=False, regex=False)
                    | results.genres.str.contains(query, case=False, regex=False))
            results = results[mask]
        return results.sort_values(["rating_count", "movieId"], ascending=[False, True]).head(limit).copy()

    def _validate_ratings(self, user_ratings):
        if not isinstance(user_ratings, Mapping):
            raise PreferenceError("Ratings must map movie IDs to half-star ratings.")
        cleaned = {}
        for raw_id, rating in user_ratings.items():
            movie_id = _integer_id(raw_id)
            if movie_id not in self._positions:
                raise PreferenceError(f"Movie ID {movie_id} is not in the movie catalog.")
            if (isinstance(rating, bool) or not isinstance(rating, Real)
                    or not np.isfinite(rating) or not 0.5 <= rating <= 5.0
                    or not np.isclose(rating * 2, round(rating * 2))):
                raise PreferenceError("Choose ratings from 0.5 to 5.0 in half-star steps.")
            cleaned[movie_id] = float(rating)
        return cleaned

    def recommend_for_new_user(self, favorite_genres, user_ratings, top_n=10,
                               content_weight=0.5, embedding_weight=0.5,
                               min_rating_count=0):
        """Return ranking scores (not predicted stars) plus deterministic explanations.

        Ratings shape separate genre and movie-embedding profiles using (rating - 3).
        The ratings score blends their available cosine similarities. Explicit genres
        form an independent score with a 25% share when ratings also guide the ranking.
        Cosines in [-1, 1] map to [0, 1] with (cosine + 1) / 2.
        Rating count only breaks exact score ties, unless an explicit filter is set.
        """
        if isinstance(favorite_genres, str):
            raise PreferenceError("Choose genres as a list, not a single text string.")
        genres = set(favorite_genres or [])
        if genres - set(self.genres):
            raise PreferenceError("Choose genres available in the movie catalog.")
        ratings = self._validate_ratings(user_ratings)
        for value, label in [(top_n, "top_n"), (min_rating_count, "min_rating_count")]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (1 if label == "top_n" else 0):
                raise PreferenceError(f"{label} must be a valid integer.")
        try:
            weights = np.asarray([content_weight, embedding_weight], dtype=float)
        except (TypeError, ValueError) as error:
            raise PreferenceError("Recommendation weights must be numbers.") from error
        if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise PreferenceError("Use finite nonnegative weights with a positive total.")
        weights /= weights.sum()
        favorite_profile = np.zeros(len(self.genres))
        for genre in genres:
            favorite_profile[self._genre_columns[genre]] = 1.0
        embedding_profile = np.zeros(self._embedding_units.shape[1])
        content_profile = np.zeros(len(self.genres))
        for movie_id, rating in ratings.items():
            row, weight = self._positions[movie_id], rating - 3.0
            content_profile += weight * self._genre_units[row]
            embedding_profile += weight * self._embedding_units[row]
        # Cosine normalization makes division by total absolute rating weight unnecessary
        # now that the ratings profile is independent of the favorite-genre profile.
        has_content = np.linalg.norm(content_profile) > 1e-12
        has_embedding = np.linalg.norm(embedding_profile) > 1e-12
        if not genres and not has_content and not has_embedding:
            raise PreferenceError("Choose a genre or add a rating above or below 3 stars so we can learn your taste.")

        content_raw = np.zeros(len(self._movies))
        embedding_raw = np.zeros(len(self._movies))
        if has_content:
            content_raw = self._genre_units @ (content_profile / np.linalg.norm(content_profile))
        if has_embedding:
            embedding_raw = self._embedding_units @ (embedding_profile / np.linalg.norm(embedding_profile))
        has_genres = self._genre_features.any(axis=1)
        content_available = has_content & has_genres
        favorite_available = bool(genres and weights[0] > 0) & has_genres
        favorite_score = np.zeros(len(self._movies))
        if favorite_available.any():
            favorite_raw = self._genre_units @ (favorite_profile / np.linalg.norm(favorite_profile))
            favorite_score = np.clip((favorite_raw + 1) / 2, 0, 1)
        embedding_available = has_embedding & self._has_embedding
        content_score = np.clip((content_raw + 1) / 2, 0, 1)
        embedding_score = np.clip((embedding_raw + 1) / 2, 0, 1)
        denominator = weights[0] * content_available + weights[1] * embedding_available
        ratings_available = denominator > 0
        if not (ratings_available | favorite_available).any():
            raise PreferenceError("These weights disable your available preferences; allow a content or movie-history signal.")
        numerator = (weights[0] * content_available * content_score
                     + weights[1] * embedding_available * embedding_score)
        ratings_score = np.divide(numerator, denominator, out=np.full_like(numerator, 0.5),
                                  where=ratings_available)
        # A title with only a favorite-genre score gets a neutral ratings contribution
        # when the request otherwise has usable rating signals.
        with_favorites = (FAVORITE_GENRE_WEIGHT * favorite_score
                          + (1 - FAVORITE_GENRE_WEIGHT) * ratings_score
                          if ratings_available.any() else favorite_score)
        scores = np.where(favorite_available, with_favorites, ratings_score)
        results = self._movies.copy()
        results["content_score"] = np.where(content_available, content_score,
                                            np.where(favorite_available, favorite_score, np.nan))
        results["favorite_genre_score"] = np.where(favorite_available, favorite_score, np.nan)
        results["embedding_score"] = np.where(embedding_available, embedding_score, np.nan)
        results["hybrid_score"] = scores
        eligible = ((ratings_available | favorite_available) & ~results.movieId.isin(ratings).to_numpy()
                    & (results.rating_count.to_numpy() >= min_rating_count))
        results = results[eligible].sort_values(
            ["hybrid_score", "rating_count", "movieId"], ascending=[False, False, True]
        ).head(top_n).copy()
        liked_with_embeddings = [(self._positions[mid], rating) for mid, rating in ratings.items()
                                 if rating > 3 and self._has_embedding[self._positions[mid]]]
        explanations = []
        for movie_id in results.movieId:
            row = self._positions[movie_id]
            movie_genres = self._genre_lists.iloc[row]
            selected = [g for g in movie_genres if g in genres]
            learned = [g for g in movie_genres
                       if content_profile[self._genre_columns[g]] > 1e-12]
            parts = []
            if favorite_available[row] and selected:
                parts.append("You selected " + " and ".join(selected[:2]) + " as favorite genres.")
            if weights[0] > 0 and content_raw[row] > 1e-12 and learned:
                parts.append("Your ratings favor " + " and ".join(learned[:2]) + " movies.")
            if weights[1] > 0 and embedding_available[row] and embedding_raw[row] > 1e-12:
                similar = [(float(self._embedding_units[row] @ self._embedding_units[liked_row]),
                            rating, liked_row) for liked_row, rating in liked_with_embeddings]
                if similar and max(similar)[0] > 1e-12:
                    _, rating, liked_row = max(similar)
                    parts.append(f"MovieLens rating patterns are similar to those for "
                                 f"{self._movies.title.iloc[liked_row]}, which you rated {rating:g}/5.")
                else:
                    parts.append("It fits the broader MovieLens patterns from your ratings.")
            explanations.append(" ".join(parts) or "This is a weaker match for your overall preferences.")
        results["explanation"] = explanations
        return results.reset_index(drop=True)

    def recommend_for_known_user(self, user_id, top_n=10):
        """Preserve trained-user inference; return clipped predicted star ratings."""
        user_id = _integer_id(user_id)
        if self._model is None or user_id not in self._user_to_index:
            raise PreferenceError("This user has no trained embedding. Use the new-user flow instead.")
        if isinstance(top_n, bool) or not isinstance(top_n, Integral) or top_n < 1:
            raise PreferenceError("top_n must be a positive integer.")
        seen = set(self._ratings.loc[self._ratings.userId == user_id, "movieId"])
        candidates = self._movies[self._movies.movieId.isin(self._movie_to_index)
                                  & ~self._movies.movieId.isin(seen)].copy()
        predictions = []
        with self._model_lock:
            for start in range(0, len(candidates), 256):
                batch = candidates.iloc[start:start + 256]
                inputs = {"user": np.full((len(batch), 1), self._user_to_index[user_id], dtype="int32"),
                          "movie": batch.movieId.map(self._movie_to_index).to_numpy(dtype="int32").reshape(-1, 1)}
                predictions.extend(np.asarray(self._model(inputs, training=False)).ravel())
        candidates["predicted_rating"] = np.clip(predictions, 0.5, 5.0)
        return candidates.sort_values(["predicted_rating", "movieId"], ascending=[False, True]).head(top_n).reset_index(drop=True)
