"""Small synthetic tests isolate scoring behavior; no training required."""
import unittest
import numpy as np
import pandas as pd
from src.hybrid_recommender import HybridRecommender, PreferenceError


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame([
            (10, "Duplicate (2000)", "Action"),
            (20, "Duplicate (2000)", "Comedy"),
            (30, "Action match (2001)", "Action"),
            (40, "Comedy match (2001)", "Comedy"),
            (50, "Both (2002)", "Action|Comedy"),
            (60, "No factors (2003)", "Action"),
            (70, "No genres (2004)", "(no genres listed)"),
        ], columns=["movieId", "title", "genres"])
        # Intentionally shuffled mapping: tests must not rely on CSV row order.
        mapping = {20: 0, 10: 1, 40: 2, 30: 3, 50: 4, 70: 5}
        factors = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 1], [-1, 0]], dtype=float)
        self.engine = HybridRecommender(self.movies, mapping, factors)

    def test_genres_only(self):
        result = self.engine.recommend_for_new_user(["Action"], {}, top_n=3)
        self.assertEqual(result.movieId.tolist(), [10, 30, 60])
        self.assertTrue(result.embedding_score.isna().all())
        self.assertTrue(result.content_score.between(0, 1).all())

    def test_ratings_only_and_exclude_seen(self):
        result = self.engine.recommend_for_new_user([], {10: 5}, top_n=5)
        self.assertNotIn(10, result.movieId.values)
        self.assertEqual(result.iloc[0].movieId, 30)
        self.assertEqual(result.iloc[0].embedding_score, 1)

    def test_hybrid_sorted_top_n_and_non_normalized_weights(self):
        a = self.engine.recommend_for_new_user(["Comedy"], {10: 4.5}, top_n=4, content_weight=5, embedding_weight=5)
        b = self.engine.recommend_for_new_user(["Comedy"], {10: 4.5}, top_n=4)
        self.assertEqual(len(a), 4)
        self.assertTrue(a.hybrid_score.is_monotonic_decreasing)
        self.assertTrue(a.hybrid_score.between(0, 1).all())
        pd.testing.assert_frame_equal(a, b)

    def test_favorite_genres_have_a_separate_bounded_share(self):
        ratings = {10: 5}
        only_ratings = self.engine.recommend_for_new_user([], ratings).set_index("movieId")
        mixed = self.engine.recommend_for_new_user(["Comedy"], ratings).set_index("movieId")
        self.assertTrue(only_ratings.favorite_genre_score.isna().all())
        for movie_id in [30, 40, 50, 60]:
            self.assertAlmostEqual(
                mixed.loc[movie_id, "hybrid_score"],
                0.75 * only_ratings.loc[movie_id, "hybrid_score"]
                + 0.25 * mixed.loc[movie_id, "favorite_genre_score"],
            )
        self.assertGreater(mixed.loc[30, "hybrid_score"], mixed.loc[40, "hybrid_score"])
        self.assertIn("You selected Comedy", mixed.loc[40, "explanation"])

    def test_dislike_reverses_preference(self):
        liked = self.engine.recommend_for_new_user([], {10: 5}, top_n=6).set_index("movieId")
        disliked = self.engine.recommend_for_new_user([], {10: 0.5}, top_n=6).set_index("movieId")
        self.assertGreater(liked.loc[30, "hybrid_score"], liked.loc[40, "hybrid_score"])
        self.assertLess(disliked.loc[30, "hybrid_score"], disliked.loc[40, "hybrid_score"])

    def test_invalid_ids_and_ratings(self):
        for ratings in [{999: 5}, {10.5: 4}, {True: 4}, {10: 0}, {10: 5.5}, {10: 3.2}, {10: np.nan}, {10: "5"}]:
            with self.subTest(ratings=ratings), self.assertRaises(PreferenceError):
                self.engine.recommend_for_new_user([], ratings)

    def test_no_useful_preferences(self):
        for genres, ratings in [([], {}), ([], {10: 3}), (["Unknown"], {})]:
            with self.assertRaises(PreferenceError):
                self.engine.recommend_for_new_user(genres, ratings)

    def test_missing_embedding_uses_content(self):
        result = self.engine.recommend_for_new_user(["Action"], {10: 5}, top_n=6).set_index("movieId")
        self.assertTrue(np.isnan(result.loc[60, "embedding_score"]))
        self.assertEqual(result.loc[60, "hybrid_score"], result.loc[60, "content_score"])
        from_missing = self.engine.recommend_for_new_user([], {60: 5}, top_n=2)
        self.assertNotIn(60, from_missing.movieId.values)
        self.assertTrue(from_missing.embedding_score.isna().all())

    def test_duplicate_titles_stay_separate(self):
        self.assertEqual(len(self.engine.search_movies("Duplicate")), 2)
        result = self.engine.recommend_for_new_user(["Comedy"], {10: 5}, top_n=6)
        self.assertIn(20, result.movieId.values)
        self.assertNotIn(10, result.movieId.values)

    def test_missing_genres_and_empty_candidates(self):
        result = self.engine.recommend_for_new_user(["Action"], {}, top_n=20)
        self.assertNotIn(70, result.movieId.values)
        seen = {movie_id: 4 for movie_id in self.movies.movieId}
        self.assertTrue(self.engine.recommend_for_new_user(["Comedy"], seen).empty)

    def test_invalid_weights_and_limits(self):
        for kwargs in [dict(content_weight=-1), dict(content_weight=np.inf),
                       dict(content_weight=0, embedding_weight=0), dict(top_n=0), dict(top_n=2.2)]:
            with self.assertRaises(PreferenceError):
                self.engine.recommend_for_new_user(["Action"], {}, **kwargs)

    def test_explanations_and_copy_safety(self):
        result = self.engine.recommend_for_new_user(["Action"], {}, top_n=1)
        self.assertIn("Action", result.iloc[0].explanation)
        from_rating = self.engine.recommend_for_new_user([], {10: 4.5}, top_n=1).iloc[0]
        self.assertIn("Your ratings favor Action", from_rating.explanation)
        self.assertIn("Duplicate (2000), which you rated 4.5/5", from_rating.explanation)
        content_only = self.engine.recommend_for_new_user([], {10: 4.5}, top_n=1,
                                                          embedding_weight=0).iloc[0]
        self.assertNotIn("MovieLens rating patterns", content_only.explanation)
        movies = self.engine.movies
        movies.loc[0, "title"] = "Changed"
        self.assertNotEqual(self.engine.movies.loc[0, "title"], "Changed")


if __name__ == "__main__":
    unittest.main()
