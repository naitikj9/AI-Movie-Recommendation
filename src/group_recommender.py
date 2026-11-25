"""Combine several people's new-user rankings into one list for the group."""
from __future__ import annotations

import pandas as pd

from src.hybrid_recommender import PreferenceError

# "average" favours the overall group; "least_misery" protects the least happy person.
STRATEGIES = ("average", "least_misery")
MIN_MEMBERS, MAX_MEMBERS = 2, 3


def recommend_for_group(engine, members, top_n=6, strategy="average", popularity_weight=0.0,
                        allowed_ids=None):
    """Rank movies for a group.

    ``members`` is a list of dicts with ``name``, ``genres`` and ``ratings``.
    Raw scores are not comparable between people (a genre-only profile and a
    ratings profile spread differently), so each person's scores are first turned
    into percentiles among the movies that can be scored for everyone.
    A movie rated by any member is excluded because someone has already seen it.
    """
    if strategy not in STRATEGIES:
        raise PreferenceError("Choose a valid group strategy.")
    if not MIN_MEMBERS <= len(members) <= MAX_MEMBERS:
        raise PreferenceError(f"A group needs {MIN_MEMBERS} to {MAX_MEMBERS} people.")
    names = [str(member["name"]).strip() for member in members]
    if not all(names) or len({name.casefold() for name in names}) != len(names):
        raise PreferenceError("Give each person a different name.")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise PreferenceError("top_n must be a positive integer.")

    catalog = engine.movies.set_index("movieId", drop=False)
    seen = {int(movie_id) for member in members for movie_id in member["ratings"]}
    scores = []
    for name, member in zip(names, members):
        try:
            ranking = engine.recommend_for_new_user(
                member["genres"], member["ratings"], top_n=len(catalog), explain=False,
                popularity_weight=popularity_weight, allowed_ids=allowed_ids,
            )
        except PreferenceError as error:
            raise PreferenceError(f"{name}: {error}") from error
        scores.append(ranking.set_index("movieId").hybrid_score)

    shared = pd.concat(scores, axis=1, join="inner", keys=range(len(names)))
    shared = shared[~shared.index.isin(seen)]
    percentiles = shared.rank(pct=True)
    group_score = percentiles.mean(axis=1) if strategy == "average" else percentiles.min(axis=1)

    columns = ["movieId", "title", "genres", "rating_count"]
    results = catalog.loc[shared.index, columns].rename_axis(None)
    results["group_score"] = group_score
    results["average_score"] = percentiles.mean(axis=1)
    results["member_scores"] = pd.Series(
        [dict(zip(names, row)) for row in percentiles.itertuples(index=False, name=None)],
        index=percentiles.index, dtype=object,
    )
    # Average breaks least-misery ties; popularity only breaks remaining exact ties.
    return results.sort_values(
        ["group_score", "average_score", "rating_count", "movieId"],
        ascending=[False, False, False, True],
    ).head(top_n).reset_index(drop=True)
