"""Exercise Streamlit widgets with the Bollywood catalog."""
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SKIPPED = "Haven’t Seen"
FIRST = 229633  # 3 Idiots
OTHER = 949228  # Sholay


class AppFlowTests(unittest.TestCase):
    def setUp(self):
        poster = patch("src.posters.fetch_poster", return_value=None)
        poster.start()
        self.addCleanup(poster.stop)

    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        return app

    def test_genre_navigation_does_not_load_model(self):
        with patch("src.hybrid_recommender.HybridRecommender.from_bollywood") as load_engine:
            app = self.app()
            app.button(key="choose_genres").click().run()
            self.assertTrue(app.button(key="continue").disabled)
            app.pills(key="genre_selection").set_value(["Action", "Sci-Fi"]).run()
            app.button(key="continue").click().run()
            self.assertEqual(app.session_state["favorite_genres"], ["Action", "Sci-Fi"])
            app.button(key="back").click().run()
            self.assertEqual(app.pills(key="genre_selection").value, ["Action", "Sci-Fi"])
            app.pills(key="genre_selection").set_value(["Sci-Fi"]).run()
            self.assertEqual(app.session_state["favorite_genres"], ["Sci-Fi"])
            self.assertFalse(app.exception)
            load_engine.assert_not_called()

    def test_choose_edit_skip_and_remove_movie(self):
        app = self.app()
        app.button(key="choose_genres").click().run()
        app.pills(key="genre_selection").select("Action").run()
        app.button(key="continue").click().run()
        picker = app.selectbox(key="movie_picker")
        self.assertIn("3 Idiots (2009)", picker.options)
        self.assertGreater(len(picker.options), 3000)
        picker.select(FIRST).run()
        preview = next(item.value for item in app.get("html") if 'class="search-preview"' in item.value)
        self.assertIn("Poster unavailable", preview)
        self.assertIn("3 Idiots (2009)", preview)
        app.selectbox(key=f"rate_{FIRST}").select(SKIPPED).run()
        self.assertEqual(app.session_state["skipped_movies"], {FIRST})
        self.assertEqual(app.session_state["user_ratings"], {})
        app.selectbox(key="movie_picker").select(OTHER).run()
        app.selectbox(key="movie_picker").select(FIRST).run()
        self.assertEqual(app.selectbox(key=f"rate_{FIRST}").value, SKIPPED)
        app.selectbox(key=f"rate_{FIRST}").select(4.5).run()
        self.assertEqual(app.session_state["user_ratings"], {FIRST: 4.5})
        self.assertEqual(app.session_state["skipped_movies"], set())
        app.selectbox(key=f"edit_{FIRST}").select(SKIPPED).run()
        self.assertEqual(app.selectbox(key=f"rate_{FIRST}").value, SKIPPED)
        self.assertEqual(app.session_state["user_ratings"], {})
        app.selectbox(key=f"rate_{FIRST}").select(2.0).run()
        app.button(key=f"remove_{FIRST}").click().run()
        self.assertEqual(app.session_state["user_ratings"], {})
        self.assertIsNone(app.selectbox(key=f"rate_{FIRST}").value)
        app.button(key="get_recommendations").click().run()
        self.assertFalse(app.exception)
        markup = " ".join(item.value for item in app.get("html"))
        self.assertIn("Why this movie?", markup)
        self.assertIn("Poster unavailable", markup)
        self.assertIn("You selected Action", markup)

    def test_ratings_only_requires_five_and_handles_neutral_profile(self):
        app = self.app()
        app.button(key="choose_movies").click().run()
        self.assertTrue(app.button(key="get_recommendations").disabled)
        for index, (movie_id, rating) in enumerate(
            [(229633, 5.0), (949228, 4.5), (20762668, 4.0), (1759650, 1.0), (843949, 5.0)], 1
        ):
            app.selectbox(key="movie_picker").select(movie_id).run()
            app.selectbox(key=f"rate_{movie_id}").select(rating).run()
            self.assertEqual(app.button(key="get_recommendations").disabled, index < 5)
        app.button(key="get_recommendations").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("Why this movie?", " ".join(item.value for item in app.get("html")))
        app.button(key="rate_more").click().run()
        for movie_id in app.session_state["user_ratings"]:
            app.selectbox(key=f"edit_{movie_id}").select(3.0).run()
        app.button(key="get_recommendations").click().run()
        self.assertTrue(any("above or below 3" in item.value for item in app.info))
        self.assertFalse(app.exception)

    def test_switch_modes_uses_only_relevant_inputs(self):
        app = self.app()
        self.assertIn("How would you like to", " ".join(item.value for item in app.get("html")))
        ratings = {229633: 5.0, 949228: 4.5, 20762668: 4.0, 1759650: 1.0, 843949: 5.0}
        app.session_state["user_ratings"] = ratings
        app.button(key="choose_genres").click().run()
        app.pills(key="genre_selection").select("Action").run()
        app.button(key="continue").click().run()
        app.button(key="get_recommendations").click().run()
        profile, results = app.session_state["recommendation_result"]
        self.assertEqual(profile, ("genres", ("Action",), tuple(sorted(ratings.items())), "All years"))
        self.assertTrue(results.favorite_genre_score.notna().any())

        app.button(key="update_preferences").click().run()
        app.button(key="choose_movies").click().run()
        self.assertEqual(app.session_state["favorite_genres"], [])
        self.assertEqual(app.session_state["user_ratings"], ratings)
        app.session_state["favorite_genres"] = ["Sci-Fi"]
        app.button(key="get_recommendations").click().run()
        profile, results = app.session_state["recommendation_result"]
        self.assertEqual(profile, ("movies", (), tuple(sorted(ratings.items())), "All years"))
        self.assertTrue(results.favorite_genre_score.isna().all())

        app.button(key="update_preferences").click().run()
        app.button(key="choose_genres").click().run()
        self.assertEqual(app.pills(key="genre_selection").value, [])
        self.assertTrue(app.button(key="continue").disabled)

    def test_group_flow_shows_each_persons_match(self):
        app = self.app()
        app.button(key="choose_group").click().run()
        self.assertTrue(app.button(key="group_submit").disabled)
        app.multiselect(key="group_genres_0").select("Action").run()
        app.multiselect(key="group_genres_1").select("Comedy").run()
        app.button(key="group_submit").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        _, results = app.session_state["group_result"]
        self.assertEqual(len(results), 6)
        markup = " ".join(item.value for item in app.get("html"))
        self.assertIn("Match for each person", markup)
        self.assertIn("Person 2", markup)

    def test_mood_fills_genres_and_manual_change_clears_it(self):
        app = self.app()
        app.button(key="choose_genres").click().run()
        app.pills(key="mood_selection").set_value("Romantic").run()
        self.assertEqual(app.session_state["favorite_genres"], ["Musical", "Romance"])
        self.assertEqual(app.pills(key="genre_selection").value, ["Musical", "Romance"])
        self.assertFalse(app.button(key="continue").disabled)
        app.pills(key="genre_selection").set_value(["Romance"]).run()
        self.assertIsNone(app.pills(key="mood_selection").value)
        self.assertEqual(app.session_state["favorite_genres"], ["Romance"])
        self.assertFalse(app.exception)

    def test_era_filter_and_watchlist(self):
        app = self.app()
        app.button(key="choose_genres").click().run()
        app.pills(key="genre_selection").select("Action").run()
        app.button(key="continue").click().run()
        app.button(key="get_recommendations").click().run()
        app.selectbox(key="result_era").select("90s").run()
        _, results = app.session_state["recommendation_result"]
        self.assertEqual(len(results), 6)
        self.assertTrue(results.title.str.contains(r"\(199\d\)$").all())
        movie_id = int(results.movieId.iloc[0])
        app.button(key=f"result_watch_{movie_id}").click().run()
        self.assertEqual(app.session_state["watchlist"], [movie_id])
        self.assertIn("(1)", app.button(key="open_watchlist").label)
        app.button(key="open_watchlist").click().run()
        self.assertIn(results.title.iloc[0], [item.value for item in app.text])
        app.button(key="watchlist_back").click().run()
        self.assertEqual(app.selectbox(key="result_era").value, "90s")
        app.button(key="open_watchlist").click().run()
        app.button(key=f"list_watch_{movie_id}").click().run()
        self.assertEqual(app.session_state["watchlist"], [])
        self.assertIn("Nothing saved yet", app.info[0].value)
        self.assertFalse(app.exception)

    def test_group_inputs_survive_a_watchlist_visit(self):
        app = self.app()
        app.button(key="choose_group").click().run()
        app.multiselect(key="group_genres_0").select("Action").run()
        app.multiselect(key="group_genres_1").select("Comedy").run()
        app.selectbox(key="group_era").select("2010s").run()
        app.button(key="group_submit").click().run()
        _, results = app.session_state["group_result"]
        self.assertTrue(results.title.str.contains(r"\(201\d\)$").all())
        app.button(key="open_watchlist").click().run()
        app.button(key="watchlist_back").click().run()
        self.assertEqual(app.multiselect(key="group_genres_0").value, ["Action"])
        self.assertEqual(app.selectbox(key="group_era").value, "2010s")
        self.assertIn("Match for each person", " ".join(item.value for item in app.get("html")))
        self.assertFalse(app.exception)

    def test_invalid_retained_movie_id_can_be_removed(self):
        app = self.app()
        app.button(key="choose_genres").click().run()
        app.session_state["user_ratings"] = {999999: 5.0}
        app.pills(key="genre_selection").select("Action").run()
        app.button(key="continue").click().run()
        self.assertIn("unavailable", app.warning[0].value)
        app.button(key="get_recommendations").click().run()
        self.assertIn("999999 is not in the movie catalog", app.info[0].value)
        app.button(key="rate_more").click().run()
        app.button(key="remove_999999").click().run()
        self.assertEqual(app.session_state["user_ratings"], {})
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
