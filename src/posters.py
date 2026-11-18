"""Optional TMDB poster lookup. Never sends user preferences or model data."""
import base64
from io import BytesIO
import re

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError


def load_tmdb_mapping(path):
    """MovieLens movieId -> TMDB movie ID; skip missing/unusable links."""
    try:
        links = pd.read_csv(path, usecols=["movieId", "tmdbId"])
    except (OSError, ValueError):
        return {}
    mapping = {}
    for movie_id, tmdb_id in links.itertuples(index=False, name=None):
        if pd.notna(movie_id) and pd.notna(tmdb_id):
            try:
                if float(tmdb_id).is_integer() and float(movie_id).is_integer() and tmdb_id > 0:
                    mapping[int(movie_id)] = int(tmdb_id)
            except (TypeError, ValueError, OverflowError):
                continue
    return mapping


def fetch_poster(tmdb_id, api_key="", read_token="", *, http_get=requests.get):
    """Return verified image data URI, or None for any unavailable poster.

    Credentials remain server-side. Only the TMDB ID is looked up (no title
    guessing, which would confuse duplicate titles). Callers cache results.
    """
    if not api_key and not read_token:
        return None
    if not isinstance(tmdb_id, int) or isinstance(tmdb_id, bool) or tmdb_id <= 0:
        return None
    headers = {"Authorization": f"Bearer {read_token}"} if read_token else {}
    params = {} if read_token else {"api_key": api_key}
    try:
        response = http_get(f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                            headers=headers, params=params, timeout=(2, 4))
        response.raise_for_status()
        details = response.json()
        path = details.get("poster_path") if isinstance(details, dict) else None
        if not isinstance(path, str) or not re.fullmatch(r"/[A-Za-z0-9_-]+\.(?:jpg|png|webp)", path):
            return None
        image_response = http_get("https://image.tmdb.org/t/p/w500" + path, timeout=(2, 4))
        image_response.raise_for_status()
        content = image_response.content
        if not content or len(content) > 5_000_000:
            return None
        with Image.open(BytesIO(content)) as image:
            mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(image.format)
            image.verify()
        if not mime:
            return None
        return f"data:{mime};base64," + base64.b64encode(content).decode("ascii")
    except (requests.RequestException, ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        # Authentication errors, rate limits, missing images, and offline use all
        # retain the existing placeholder. Never expose credential-bearing errors.
        return None
