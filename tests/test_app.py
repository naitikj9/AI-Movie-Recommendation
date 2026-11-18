"""Exercise real Streamlit widgets and artifacts; never train the network."""
from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class AppFlowTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        return app

    def test_full_flow_persistence_edit_remove_and_search(self):
        app = self.app()
        app.button(key="genre_Action").click().run()
        app.button(key="genre_Sci-Fi").click().run()
        app.button(key="continue").click().run()
        self.assertEqual(app.session_state["favorite_genres"], ["Action", "Sci-Fi"])
        app.text_input(key="movie_search").input("Toy Story (1995)").run()
        self.assertFalse(app.exception)
        app.selectbox(key="rate_1").select(4.5).run()
        self.assertEqual(app.session_state["user_ratings"], {1: 4.5})
        app.selectbox(key="edit_1").select(2.0).run()
        self.assertEqual(app.session_state["user_ratings"][1], 2.0)
        app.button(key="get_recommendations").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["step"], 3)
        # HTML cards contain explanations, not technical scores or star predictions.
        self.assertTrue(any("Your Recommendations" in item.value for item in app.title))
        app.button(key="rate_more").click().run()
        self.assertEqual(app.session_state["user_ratings"], {1: 2.0})
        self.assertEqual(app.selectbox(key="edit_1").value, 2.0)
        app.button(key="remove_1").click().run()
        self.assertEqual(app.session_state["user_ratings"], {})
        self.assertIsNone(app.selectbox(key="rate_1").value)
        app.text_input(key="movie_search").input("zzzz_movie_not_found").run()
        self.assertTrue(any("No movies found" in item.value for item in app.info))
        app.button(key="back").click().run()
        self.assertEqual(app.session_state["favorite_genres"], ["Action", "Sci-Fi"])
        app.button(key="genre_Action").click().run()
        self.assertEqual(app.session_state["favorite_genres"], ["Sci-Fi"])
        self.assertFalse(app.exception)

    def test_ratings_only_requires_five_and_handles_neutral_profiles(self):
        app = self.app()
        app.button(key="continue").click().run()
        self.assertTrue(app.button(key="get_recommendations").disabled)
        for movie_id, rating in [(356, 5.0), (318, 4.5), (296, 4.0), (593, 1.0), (2571, 5.0)]:
            app.selectbox(key=f"rate_{movie_id}").select(rating).run()
        self.assertFalse(app.button(key="get_recommendations").disabled)
        app.button(key="get_recommendations").click().run()
        self.assertFalse(app.exception)
        app.button(key="rate_more").click().run()
        for movie_id in list(app.session_state["user_ratings"]):
            app.selectbox(key=f"edit_{movie_id}").select(3.0).run()
        app.button(key="get_recommendations").click().run()
        self.assertTrue(any("above or below 3" in item.value for item in app.info))
        self.assertFalse(app.exception)


if __name__ == "__main__":
    unittest.main()
