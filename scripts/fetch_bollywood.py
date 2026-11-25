"""Build data/bollywood/movies.csv from Wikidata (CC0).

Usage:  python scripts/fetch_bollywood.py [--raw path/to/saved_response.json]

Selects films whose original language is Hindi and whose country of origin is India.
That is a close proxy for "Bollywood", not an exact industry label.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/bollywood/movies.csv"
ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
SELECT ?film (SAMPLE(?label) AS ?title) (MIN(YEAR(?date)) AS ?year)
       (GROUP_CONCAT(DISTINCT ?genreLabel; separator="|") AS ?genres)
       (SAMPLE(?links) AS ?sitelinks) (SAMPLE(?tmdb) AS ?tmdbId)
WHERE {
  ?film wdt:P31 wd:Q11424 ;
        wdt:P364 wd:Q1568 ;
        wdt:P495 wd:Q668 ;
        wdt:P577 ?date ;
        wdt:P136 ?genre ;
        wikibase:sitelinks ?links ;
        rdfs:label ?label .
  FILTER(LANG(?label) = "en")
  ?genre rdfs:label ?genreLabel .
  FILTER(LANG(?genreLabel) = "en")
  OPTIONAL { ?film wdt:P4947 ?tmdb . }
}
GROUP BY ?film
"""

# Wikidata has hundreds of fine-grained genres; group them into the ones the app shows.
GENRE_KEYWORDS = {
    "Action": ("action", "martial arts", "masala", "superhero"),
    "Adventure": ("adventure",),
    "Animation": ("animat",),
    "Biography": ("biograph", "biopic"),
    "Comedy": ("comedy", "satir", "parody"),
    "Crime": ("crime", "gangster", "heist", "police"),
    "Drama": ("drama", "melodrama"),
    "Family": ("family", "children"),
    "Fantasy": ("fantasy", "mytholog", "supernatural"),
    "History": ("histor", "period"),
    "Horror": ("horror",),
    "Musical": ("musical", "music film", "dance"),
    "Mystery": ("mystery", "detective"),
    "Romance": ("roman", "love"),
    "Sci-Fi": ("science fiction",),
    "Sports": ("sport",),
    "Thriller": ("thriller", "suspense"),
    "War": ("war film", "war drama", "anti-war"),
}


def clean_genres(raw):
    found = set()
    for name in raw.lower().split("|"):
        # "dramedy" and "comedy drama" contain two genres, so test every group.
        found.update(genre for genre, words in GENRE_KEYWORDS.items()
                     if any(word in name for word in words))
    return sorted(found)


def download():
    response = requests.get(
        ENDPOINT, params={"query": QUERY, "format": "json"}, timeout=120,
        headers={"User-Agent": "bollywood-catalog-script/1.0",
                 "Accept": "application/sparql-results+json"},
    )
    response.raise_for_status()
    return response.content


def build_rows(payload):
    rows, seen = [], set()
    for item in payload["results"]["bindings"]:
        value = {key: field["value"] for key, field in item.items()}
        genres = clean_genres(value.get("genres", ""))
        match = re.search(r"Q(\d+)$", value["film"])
        if not genres or not match or "year" not in value or match.group(1) in seen:
            continue
        seen.add(match.group(1))
        tmdb = value.get("tmdbId", "")
        rows.append({
            "movieId": int(match.group(1)),
            "title": f"{value['title']} ({value['year']})",
            "genres": "|".join(genres),
            "sitelinks": int(value["sitelinks"]),
            "tmdbId": tmdb if tmdb.isdigit() else "",
        })
    return sorted(rows, key=lambda row: row["movieId"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="reuse a saved Wikidata JSON response")
    arguments = parser.parse_args()
    content = arguments.raw.read_bytes() if arguments.raw else download()
    print(f"Wikidata response: {len(content) / 1024:.0f} KB")
    rows = build_rows(json.loads(content))
    if not rows:
        raise SystemExit("No films returned; the existing catalog was left unchanged.")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} films to {OUTPUT.relative_to(ROOT)} ({OUTPUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
