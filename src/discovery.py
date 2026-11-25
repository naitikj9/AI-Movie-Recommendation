"""Era filter, mood presets, and watchlist export. No Streamlit dependency."""
import csv
from io import StringIO
import re

ALL_ERAS = "All years"
# label -> (first year, last year), inclusive. None leaves that side open.
ERAS = {
    ALL_ERAS: (None, None),
    "Before 1990": (None, 1989),
    "90s": (1990, 1999),
    "2000s": (2000, 2009),
    "2010s": (2010, 2019),
    "2020 onwards": (2020, None),
}

# A mood is a shortcut that pre-selects genres; the person can still change them.
MOODS = {
    "Family night": ("Comedy", "Drama", "Family"),
    "Romantic": ("Musical", "Romance"),
    "Edge of your seat": ("Crime", "Mystery", "Thriller"),
    "Full masala": ("Action", "Comedy", "Musical"),
    "Feel good": ("Comedy", "Romance"),
    "Inspiring": ("Biography", "Sports"),
    "Spooky": ("Horror",),
}


def release_year(title):
    match = re.search(r"\((\d{4})\)\s*$", title)
    return int(match.group(1)) if match else None


def ids_in_era(movies, era):
    """Movie IDs released in the era, or None when no filtering is needed.

    ``movies`` is a DataFrame with movieId and a title ending in "(year)".
    """
    first, last = ERAS[era]
    if first is None and last is None:
        return None
    return {
        movie_id for movie_id, title in zip(movies.movieId, movies.title)
        if (year := release_year(title)) is not None
        and (first is None or year >= first) and (last is None or year <= last)
    }


def watchlist_csv(rows):
    """CSV text for (title, genres) rows, safe to open in a spreadsheet."""
    def safe(value):
        # Spreadsheets run cells that start with these characters as formulas.
        return "'" + value if value.startswith(("=", "+", "-", "@")) else value

    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["Title", "Genres"])
    writer.writerows([safe(title), safe(genres.replace("|", ", "))] for title, genres in rows)
    return output.getvalue()
