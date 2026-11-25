"""Era filtering, mood presets, and watchlist export."""
from pathlib import Path
import unittest

from src.discovery import ALL_ERAS, ERAS, MOODS, ids_in_era, release_year, watchlist_csv
from src.hybrid_recommender import HybridRecommender

ROOT = Path(__file__).resolve().parents[1]


class DiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = HybridRecommender.from_bollywood(ROOT)

    def test_every_mood_uses_catalog_genres(self):
        for mood, genres in MOODS.items():
            self.assertTrue(set(genres) <= set(self.engine.genres), mood)

    def test_eras_cover_every_year_once(self):
        movies = self.engine.movies
        sets = [ids_in_era(movies, era) for era in ERAS if era != ALL_ERAS]
        self.assertEqual(sum(map(len, sets)), len(movies))
        self.assertEqual(set().union(*sets), set(movies.movieId))
        self.assertIsNone(ids_in_era(movies, ALL_ERAS))
        self.assertEqual(release_year("Sholay (1975)"), 1975)
        self.assertIsNone(release_year("Untitled"))

    def test_allowed_ids_limits_results(self):
        allowed = ids_in_era(self.engine.movies, "90s")
        result = self.engine.recommend_for_new_user(["Romance"], {}, top_n=10, allowed_ids=allowed)
        self.assertEqual(len(result), 10)
        self.assertTrue(set(result.movieId) <= allowed)
        self.assertTrue(self.engine.recommend_for_new_user(["Romance"], {}, allowed_ids=set()).empty)

    def test_watchlist_csv_escapes_formulas(self):
        text = watchlist_csv([("Sholay (1975)", "Action|Drama"), ("=1+1 (2000)", "Comedy")])
        self.assertEqual(text.splitlines(), [
            "Title,Genres", 'Sholay (1975),"Action, Drama"', "'=1+1 (2000),Comedy"])


if __name__ == "__main__":
    unittest.main()
