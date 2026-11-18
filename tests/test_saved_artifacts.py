"""Integration checks against the actual selected model and exact mappings."""
from pathlib import Path
import hashlib
import unittest
import numpy as np
from src.hybrid_recommender import HybridRecommender

ROOT = Path(__file__).resolve().parents[1]


class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = list((ROOT / "models").glob("*"))
        cls.before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.paths if p.is_file()}
        cls.engine = HybridRecommender.from_project(ROOT)

    def test_real_hybrid_signals_and_exclusion(self):
        for genres, ratings in [(["Action"], {}), ([], {1: 5, 2: 0.5}),
                                (["Comedy", "Sci-Fi"], {1: 4.5, 2: 1.0})]:
            result = self.engine.recommend_for_new_user(genres, ratings, top_n=10)
            self.assertEqual(len(result), 10)
            self.assertFalse(result.movieId.isin(ratings).any())
            self.assertTrue(result.hybrid_score.is_monotonic_decreasing)
            self.assertTrue(result.hybrid_score.between(0, 1).all())
            self.assertTrue(result.movieId.is_unique)
            self.assertTrue(result.explanation.str.len().gt(0).all())
        repeated = self.engine.recommend_for_new_user(["Action"], {}, top_n=10)
        original = self.engine.recommend_for_new_user(["Action"], {}, top_n=10)
        self.assertEqual(repeated.movieId.tolist(), original.movieId.tolist())

    def test_known_user_preserved(self):
        result = self.engine.recommend_for_known_user(1, top_n=10)
        self.assertEqual(len(result), 10)
        self.assertTrue(result.predicted_rating.between(0.5, 5.0).all())
        self.assertTrue(result.predicted_rating.is_monotonic_decreasing)
        seen = self.engine._ratings.loc[self.engine._ratings.userId == 1, "movieId"]
        self.assertFalse(result.movieId.isin(seen).any())

    @classmethod
    def tearDownClass(cls):
        for path, digest in cls.before.items():
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise AssertionError(f"Saved artifact modified: {path}")


if __name__ == "__main__":
    unittest.main()
