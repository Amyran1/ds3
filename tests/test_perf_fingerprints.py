from __future__ import annotations

import textwrap

from libs.autoloop.perf_fingerprints import (
    has_aggregate_in_serial_loop,
    has_iterrows_with_api_call,
    has_unbatched_embedding_loop,
)

# ---------------------------------------------------------------------------
# Detector 1: has_iterrows_with_api_call
# ---------------------------------------------------------------------------

_ITERROWS_WITH_API_CALL = textwrap.dedent(
    """\
    import pandas as pd
    import openai

    def score_rows(df):
        results = []
        for idx, row in df.iterrows():
            response = openai.embeddings.create(input=row["text"], model="text-embedding-3-small")
            results.append(response.data[0].embedding)
        return results
    """
)

_ITERROWS_NO_API_CALL = textwrap.dedent(
    """\
    import pandas as pd

    def compute_squares(df):
        results = []
        for idx, row in df.iterrows():
            results.append(row["value"] ** 2)
        return results
    """
)


def test_has_iterrows_with_api_call_true_on_matching_fixture() -> None:
    """has_iterrows_with_api_call returns True when source contains iterrows() with an API call inside the loop."""
    assert has_iterrows_with_api_call(_ITERROWS_WITH_API_CALL) is True


def test_has_iterrows_with_api_call_false_when_no_api_call() -> None:
    """has_iterrows_with_api_call returns False when iterrows() loop contains no API call."""
    assert has_iterrows_with_api_call(_ITERROWS_NO_API_CALL) is False


# ---------------------------------------------------------------------------
# Detector 2: has_aggregate_in_serial_loop
# ---------------------------------------------------------------------------

_AGGREGATE_IN_SERIAL_LOOP = textwrap.dedent(
    """\
    import pandas as pd

    def aggregate_features(df, user_ids):
        results = []
        for uid in user_ids:
            subset = df[df["user_id"] == uid]
            agg = subset.groupby("email_id").agg({"score": "mean"})
            results.append(agg)
        return pd.concat(results)
    """
)

_NO_AGGREGATE_IN_LOOP = textwrap.dedent(
    """\
    import pandas as pd

    def aggregate_features(df):
        return df.groupby("user_id").agg({"score": "mean"})
    """
)


def test_has_aggregate_in_serial_loop_true_on_matching_fixture() -> None:
    """has_aggregate_in_serial_loop returns True when a groupby/agg call appears inside a for loop."""
    assert has_aggregate_in_serial_loop(_AGGREGATE_IN_SERIAL_LOOP) is True


def test_has_aggregate_in_serial_loop_false_when_no_loop() -> None:
    """has_aggregate_in_serial_loop returns False when groupby/agg is not inside a for loop."""
    assert has_aggregate_in_serial_loop(_NO_AGGREGATE_IN_LOOP) is False


# ---------------------------------------------------------------------------
# Detector 3: has_unbatched_embedding_loop
# ---------------------------------------------------------------------------

_UNBATCHED_EMBEDDING_LOOP = textwrap.dedent(
    """\
    import openai

    def embed_texts(texts):
        embeddings = []
        for text in texts:
            resp = openai.embeddings.create(input=text, model="text-embedding-3-small")
            embeddings.append(resp.data[0].embedding)
        return embeddings
    """
)

_BATCHED_EMBEDDING_CALL = textwrap.dedent(
    """\
    import openai

    def embed_texts(texts):
        resp = openai.embeddings.create(input=texts, model="text-embedding-3-small")
        return [item.embedding for item in resp.data]
    """
)


def test_has_unbatched_embedding_loop_true_on_matching_fixture() -> None:
    """has_unbatched_embedding_loop returns True when embedding API is called inside a for loop."""
    assert has_unbatched_embedding_loop(_UNBATCHED_EMBEDDING_LOOP) is True


def test_has_unbatched_embedding_loop_false_on_batched_call() -> None:
    """has_unbatched_embedding_loop returns False when embedding call is not inside a loop."""
    assert has_unbatched_embedding_loop(_BATCHED_EMBEDDING_CALL) is False
