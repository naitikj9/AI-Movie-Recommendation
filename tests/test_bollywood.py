"""Bollywood catalog integrity, genre mapping, and popularity scoring."""
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.fetch_bollywood import build_rows, clean_genres
from src.hybrid_recommender import HybridRecommender, PreferenceError

ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = HybridRecommender.from_bollywood(ROOT)

    def test_catalog_is_complete_and_unique(self):
        movies = self.engine.movies
        self.assertGreater(len(movies), 3000)
        self.assertTrue(movies.movieId.is_unique)
        self.assertTrue(movies.genres.str.len().gt(0).all())
        self.assertIn("Sholay (1975)", set(movies.title))
        self.assertTrue(movies.popularity.between(0, 1).all())

    def test_popularity_separates_equal_genre_matches(self):
        flat = self.engine.recommend_for_new_user(["Action"], {}, top_n=6)
        self.assertEqual(flat.hybrid_score.nunique(), 1)
        ranked = self.engine.recommend_for_new_user(["Action"], {}, top_n=30, popularity_weight=0.3)
        self.assertGreater(ranked.hybrid_score.nunique(), 3)
        self.assertTrue(ranked.hybrid_score.is_monotonic_decreasing)
        self.assertTrue(ranked.hybrid_score.between(0, 1).all())
        parts = ranked[["favorite_part", "genre_part", "history_part", "neutral_part",
                        "popularity_part"]].sum(axis=1)
        np.testing.assert_allclose(parts, ranked.hybrid_score)

    def test_invalid_popularity_weight(self):
        for value in [-0.1, 1, np.nan, "0.3", True]:
            with self.subTest(value=value), self.assertRaises(PreferenceError):
                self.engine.recommend_for_new_user(["Action"], {}, popularity_weight=value)


class FetchTests(unittest.TestCase):
    def test_genre_mapping(self):
        self.assertEqual(clean_genres("action film|romantic comedy"), ["Action", "Comedy", "Romance"])
        self.assertEqual(clean_genres("unknown thing"), [])

    def test_rows_skip_unusable_and_duplicate_films(self):
        def film(qid, genre, **extra):
            values = {"film": f"http://www.wikidata.org/entity/Q{qid}", "title": "T",
                      "year": "2000", "genres": genre, "sitelinks": "3", **extra}
            return {key: {"value": value} for key, value in values.items()}

        bindings = [film(2, "drama film", tmdbId="12"), film(2, "drama film"),
                    film(1, "unknown thing"), film(3, "horror film", tmdbId="bad")]
        rows = build_rows({"results": {"bindings": bindings}})
        self.assertEqual([row["movieId"] for row in rows], [2, 3])
        self.assertEqual([row["tmdbId"] for row in rows], ["12", ""])
        self.assertEqual(rows[0]["title"], "T (2000)")


if __name__ == "__main__":
    unittest.main()
