"""Exercise Streamlit widgets with MovieLens data and the saved artifacts."""
from pathlib import Path
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SKIPPED = "Haven’t Seen"


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
        with patch("src.hybrid_recommender.HybridRecommender.from_project") as load_engine:
            app = self.app()
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
        app.pills(key="genre_selection").select("Action").run()
        app.button(key="continue").click().run()
        picker = app.selectbox(key="movie_picker")
        self.assertIn("Toy Story (1995)", picker.options)
        self.assertGreater(len(picker.options), 9000)
        picker.select(1).run()
        preview = next(item.value for item in app.get("html") if 'class="search-preview"' in item.value)
        self.assertIn("Poster unavailable", preview)
        self.assertIn("Toy Story (1995)", preview)
        app.selectbox(key="rate_1").select(SKIPPED).run()
        self.assertEqual(app.session_state["skipped_movies"], {1})
        self.assertEqual(app.session_state["user_ratings"], {})
        app.selectbox(key="movie_picker").select(318).run()
        app.selectbox(key="movie_picker").select(1).run()
        self.assertEqual(app.selectbox(key="rate_1").value, SKIPPED)
        app.selectbox(key="rate_1").select(4.5).run()
        self.assertEqual(app.session_state["user_ratings"], {1: 4.5})
        self.assertEqual(app.session_state["skipped_movies"], set())
        app.selectbox(key="edit_1").select(SKIPPED).run()
        self.assertEqual(app.selectbox(key="rate_1").value, SKIPPED)
        self.assertEqual(app.session_state["user_ratings"], {})
        app.selectbox(key="rate_1").select(2.0).run()
        app.button(key="remove_1").click().run()
        self.assertEqual(app.session_state["user_ratings"], {})
        self.assertIsNone(app.selectbox(key="rate_1").value)
        app.button(key="get_recommendations").click().run()
        self.assertFalse(app.exception)
        markup = " ".join(item.value for item in app.get("html"))
        self.assertIn("Why this movie?", markup)
        self.assertIn("Poster unavailable", markup)
        self.assertIn("You selected Action", markup)

    def test_ratings_only_requires_five_and_handles_neutral_profile(self):
        app = self.app()
        app.button(key="continue").click().run()
        self.assertTrue(app.button(key="get_recommendations").disabled)
        for index, (movie_id, rating) in enumerate(
            [(356, 5.0), (318, 4.5), (296, 4.0), (593, 1.0), (2571, 5.0)], 1
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

    def test_invalid_retained_movie_id_can_be_removed(self):
        app = self.app()
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
