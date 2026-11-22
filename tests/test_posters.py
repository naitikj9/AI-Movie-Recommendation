"""Poster tests use fake responses; they never need an API key or the network."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import requests
from PIL import Image
from src.posters import fetch_poster, load_tmdb_mapping


class PosterTests(unittest.TestCase):
    def image_bytes(self):
        buffer = BytesIO()
        Image.new("RGB", (10, 15), "red").save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_missing_credentials_does_not_call_api(self):
        get = Mock()
        self.assertIsNone(fetch_poster(862, http_get=get))
        get.assert_not_called()

    def test_real_id_mapping_skips_missing_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "links.csv"
            path.write_text("movieId,imdbId,tmdbId\n1,0114709,862\n2,0113497,\n3,0123456,123.5\n4,0555555,0\n")
            self.assertEqual(load_tmdb_mapping(path), {1: 862})
            self.assertEqual(load_tmdb_mapping(Path(directory) / "missing.csv"), {})

    def test_valid_poster_and_server_side_authentication(self):
        metadata = Mock(); metadata.json.return_value = {"poster_path": "/poster-test.jpg"}
        picture = Mock(); picture.content = self.image_bytes()
        get = Mock(side_effect=[metadata, picture])
        result = fetch_poster(862, read_token="test-token", http_get=get)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))
        self.assertEqual(get.call_args_list[0].kwargs["headers"], {"Authorization": "Bearer test-token"})
        self.assertNotIn("headers", get.call_args_list[1].kwargs)
        self.assertNotIn("test-token", result)

    def test_no_poster_bad_path_and_malformed_response(self):
        for payload in [{"poster_path": None}, {"poster_path": "https://other.invalid/a.jpg"}, [], {"poster_path": "/../../bad.jpg"}]:
            metadata = Mock(); metadata.json.return_value = payload
            get = Mock(return_value=metadata)
            self.assertIsNone(fetch_poster(862, api_key="test-key", http_get=get))
            self.assertEqual(get.call_count, 1)

    def test_authentication_rate_limit_and_network_failures(self):
        for error in [requests.Timeout(), requests.ConnectionError(), requests.HTTPError()]:
            self.assertIsNone(fetch_poster(862, api_key="test-key", http_get=Mock(side_effect=error)))

    def test_corrupt_image_uses_placeholder(self):
        metadata = Mock(); metadata.json.return_value = {"poster_path": "/image.jpg"}
        picture = Mock(); picture.content = b"not an image"
        self.assertIsNone(fetch_poster(862, api_key="test-key", http_get=Mock(side_effect=[metadata, picture])))

    def test_card_renders_verified_poster_and_escaped_title(self):
        from app import cover_html
        with patch("app.movie_poster", return_value="data:image/jpeg;base64,abc"):
            card = cover_html(1, 'A <Movie> "Title"', rank=1)
        self.assertIn('has-poster', card)
        self.assertIn('<img src="data:image/jpeg;base64,abc"', card)
        self.assertIn('&lt;Movie&gt; &quot;Title&quot;', card)
        self.assertIn('01</span>', card)

    def test_missing_poster_renders_existing_placeholder(self):
        from app import cover_html
        with patch("app.movie_poster", return_value=None):
            card = cover_html(1, "Toy Story", rank=1)
        self.assertIn('Poster unavailable', card)
        self.assertNotIn('<img', card)

    def test_search_preview_uses_poster_without_rank(self):
        from app import cover_html
        with patch("app.movie_poster", return_value="data:image/jpeg;base64,abc") as lookup:
            preview = cover_html(1, 'Toy Story (1995)')
        lookup.assert_called_once_with(1)
        self.assertIn('class="search-cover has-poster"', preview)
        self.assertIn('alt="Toy Story (1995) poster"', preview)
        self.assertNotIn('class="rank"', preview)


if __name__ == "__main__":
    unittest.main()
