"""Group ranking and score-breakdown behavior on small synthetic data."""
import unittest

import numpy as np

from src.group_recommender import recommend_for_group
from src.hybrid_recommender import PreferenceError
from tests import test_hybrid_recommender


def person(name, genres=(), ratings=None):
    return {"name": name, "genres": list(genres), "ratings": ratings or {}}


class GroupTests(unittest.TestCase):
    def setUp(self):
        fixture = test_hybrid_recommender.HybridTests()
        fixture.setUp()
        self.engine = fixture.engine

    def test_least_misery_prefers_shared_taste(self):
        members = [person("A", ["Action"]), person("B", ["Comedy"])]
        result = recommend_for_group(self.engine, members, strategy="least_misery")
        self.assertEqual(result.iloc[0].movieId, 50)
        self.assertEqual(set(result.iloc[0].member_scores), {"A", "B"})

    def test_strategy_controls_group_score(self):
        members = [person("A", ["Action"]), person("B", ["Comedy"])]
        for strategy, combine in [("average", np.mean), ("least_misery", min)]:
            for movie in recommend_for_group(self.engine, members, strategy=strategy).itertuples():
                self.assertAlmostEqual(movie.group_score, combine(list(movie.member_scores.values())))

    def test_excludes_movies_anyone_rated(self):
        members = [person("A", ratings={10: 5}), person("B", ratings={20: 5})]
        result = recommend_for_group(self.engine, members, top_n=20)
        self.assertFalse(set(result.movieId) & {10, 20})

    def test_rejects_bad_groups(self):
        good = person("A", ["Action"])
        for members in [[good], [good] * 4, [good, person("a", ["Comedy"])],
                        [good, person(" ", ["Comedy"])]]:
            with self.assertRaises(PreferenceError):
                recommend_for_group(self.engine, members)
        with self.assertRaises(PreferenceError):
            recommend_for_group(self.engine, [good, person("B", ["Comedy"])], strategy="random")
        with self.assertRaisesRegex(PreferenceError, "^B:"):
            recommend_for_group(self.engine, [good, person("B")])

    def test_breakdown_parts_add_up_to_score(self):
        for genres, ratings in [(["Action"], {}), ([], {10: 5}), (["Comedy"], {10: 4.5}),
                                (["Comedy"], {60: 5})]:
            result = self.engine.recommend_for_new_user(genres, ratings, top_n=20)
            parts = result[["favorite_part", "genre_part", "history_part", "neutral_part"]].sum(axis=1)
            np.testing.assert_allclose(parts, result.hybrid_score)


if __name__ == "__main__":
    unittest.main()
