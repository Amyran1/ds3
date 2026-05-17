from __future__ import annotations

import numpy as np
import polars as pl

_FEATURE_COLS = [
    "actioned_last_1",
    "actioned_last_3",
    "actioned_last_5",
    "actioned_last_10",
    "sends_since_last_action",
    "action_in_last_5",
    "is_in_action_streak",
    "lifetime_actions_prior",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _build_sends(
    rng: np.random.Generator,
    n_users: int,
    n_emails: int,
    include_recency: bool,
) -> pl.DataFrame:
    """Build ~50K sends for n_users across 8 quarters (2024-Q1 through 2025-Q4).

    Each user gets 10-50 sends distributed across the 24-month window.
    Per-user propensity ~ Beta(2,8); per-email quality ~ Beta(2,18).
    Recency features are computed via a forward pass over chronological history.
    """
    date_start = np.datetime64("2024-01-01", "D")
    date_end = np.datetime64("2025-12-31", "D")
    total_days = int((date_end - date_start).astype(int)) + 1

    user_propensity = rng.beta(2, 8, size=n_users)
    email_quality = rng.beta(2, 18, size=n_emails)
    sends_per_user = rng.integers(10, 51, size=n_users)

    user_id_list: list[int] = []
    email_id_list: list[int] = []
    date_ms_list: list[int] = []
    actioned_list: list[int] = []
    al1_list: list[int] = []
    al3_list: list[int] = []
    al5_list: list[int] = []
    al10_list: list[int] = []
    sla_list: list[int] = []
    ain5_list: list[int] = []
    streak_list: list[int] = []
    lt_list: list[int] = []

    _epoch = np.datetime64("1970-01-01", "D")

    for u in range(n_users):
        n_sends = int(sends_per_user[u])
        day_offsets = np.sort(rng.choice(total_days, size=n_sends, replace=False))
        email_picks = rng.integers(0, n_emails, size=n_sends)

        history: list[int] = []

        for day_off, eid in zip(day_offsets, email_picks):
            lt_actions = sum(history)
            lta1 = history[-1] if len(history) >= 1 else 0
            lta3 = sum(history[-3:]) if len(history) >= 1 else 0
            lta5 = sum(history[-5:]) if len(history) >= 1 else 0
            lta10 = sum(history[-10:]) if len(history) >= 1 else 0
            last5 = history[-5:] if len(history) >= 1 else []
            a_in_5 = 1 if any(x == 1 for x in last5) else 0
            last3 = history[-3:] if len(history) >= 3 else history
            in_streak = 1 if len(last3) == 3 and all(x == 1 for x in last3) else 0

            sla = 999
            if history and sum(history) > 0:
                for k in range(len(history) - 1, -1, -1):
                    sla = len(history) - 1 - k
                    if history[k] == 1:
                        break

            if include_recency:
                recency_signal = float(a_in_5) * 0.4 - (min(sla, 10) / 10.0) * 0.1
                noise = float(rng.normal(0, 0.3))
                p = _sigmoid(
                    np.array(
                        _logit(np.array([user_propensity[u]]))[0]
                        + 0.3 * _logit(np.array([email_quality[eid]]))[0]
                        + 0.5 * recency_signal
                        + noise
                    )
                )
            else:
                noise = float(rng.normal(0, 0.3))
                p = _sigmoid(
                    np.array(
                        _logit(np.array([user_propensity[u]]))[0]
                        + 0.3 * _logit(np.array([email_quality[eid]]))[0]
                        + noise
                    )
                )

            p_val = float(p) if hasattr(p, "__float__") else p[0]
            outcome = int(rng.random() < p_val)

            date_day = date_start + np.timedelta64(int(day_off), "D")
            date_ms = int((date_day - _epoch).astype(int)) * 86400 * 1000

            user_id_list.append(u)
            email_id_list.append(int(eid))
            date_ms_list.append(date_ms)
            actioned_list.append(outcome)
            al1_list.append(lta1)
            al3_list.append(lta3)
            al5_list.append(lta5)
            al10_list.append(lta10)
            sla_list.append(sla)
            ain5_list.append(a_in_5)
            streak_list.append(in_streak)
            lt_list.append(lt_actions)

            history.append(outcome)

    df = pl.DataFrame(
        {
            "user_id": pl.Series(user_id_list, dtype=pl.Int32),
            "email_id": pl.Series(email_id_list, dtype=pl.Int32),
            "date_sent": pl.Series(date_ms_list, dtype=pl.Int64).cast(pl.Datetime("ms")),
            "actioned_24h": pl.Series(actioned_list, dtype=pl.Int8),
            "actioned_last_1": pl.Series(al1_list, dtype=pl.Int8),
            "actioned_last_3": pl.Series(al3_list, dtype=pl.Int8),
            "actioned_last_5": pl.Series(al5_list, dtype=pl.Int8),
            "actioned_last_10": pl.Series(al10_list, dtype=pl.Int8),
            "sends_since_last_action": pl.Series(sla_list, dtype=pl.Int32),
            "action_in_last_5": pl.Series(ain5_list, dtype=pl.Int8),
            "is_in_action_streak": pl.Series(streak_list, dtype=pl.Int8),
            "lifetime_actions_prior": pl.Series(lt_list, dtype=pl.Int32),
        }
    )

    return df.sort(["user_id", "date_sent"])


def make_synthetic_user_emails(seed: int = 42) -> pl.DataFrame:
    """Build ~50K rows of synthetic user×email sends spanning 8 quarters.

    2000 users × 40 emails, 10-50 sends per user across 2024-Q1–2025-Q4.
    Outcome driven by user propensity + email quality + recency signal.
    """
    rng = np.random.default_rng(seed)
    return _build_sends(rng, n_users=2000, n_emails=40, include_recency=True)


def make_synthetic_with_interaction(seed: int = 42, n_users: int = 2000) -> pl.DataFrame:
    """Variant with a (user_id, email_id) pair random effect in the outcome logit.

    The pair effect N(0, 0.4) is NOT captured by user-only or email-only features,
    so the stricter user-main-effect residualized AUC will be > 0.50 (routing signal present).
    """
    rng = np.random.default_rng(seed)
    base = _build_sends(rng, n_users=n_users, n_emails=40, include_recency=True)

    n_emails = 40
    pair_effects = rng.normal(0.0, 0.4, size=(n_users, n_emails))

    user_ids = base["user_id"].to_numpy()
    email_ids = base["email_id"].to_numpy()
    actioned = base["actioned_24h"].to_numpy().copy().astype(np.float64)

    rng2 = np.random.default_rng(seed + 1)
    new_actioned = np.empty(len(base), dtype=np.int8)
    for i in range(len(base)):
        uid = int(user_ids[i])
        eid = int(email_ids[i])
        pair_effect = pair_effects[uid, eid]
        base_logit = _logit(np.array([actioned[i] * 0.7 + 0.15]))[0]
        p = float(_sigmoid(np.array(base_logit + pair_effect)))
        new_actioned[i] = int(rng2.random() < p)

    return base.with_columns(pl.Series("actioned_24h", new_actioned, dtype=pl.Int8))


def make_synthetic_features_equal_to_confound(seed: int = 42) -> pl.DataFrame:
    """Variant where outcome has no recency signal.

    Feature columns are still recency-based but outcome is driven purely
    by user propensity + email quality + noise. Used to test residual collapse.
    """
    rng = np.random.default_rng(seed)
    return _build_sends(rng, n_users=2000, n_emails=40, include_recency=False)


def make_sparse_confound_fixture(seed: int = 99) -> pl.DataFrame:
    """Fixture that forces one joint (user_prior, email_popularity) quartile to have < 25 positives.

    The top-user × top-email quartile is forced sparse by constraining a subset
    of high-propensity users to only interact with low-quality emails, leaving
    the top-user × top-email cell with very few positives.
    """
    rng = np.random.default_rng(seed)
    base = _build_sends(rng, n_users=2000, n_emails=40, include_recency=True)

    # Identify the top-quartile users (highest lifetime_actions_prior) and
    # top-quartile emails (email_id in top 25%). Force those user rows to use
    # only low-quality email_ids (bottom 25%) so positives in that cell are sparse.
    n_users = 2000
    n_emails = 40
    top_user_threshold = int(n_users * 0.75)
    top_email_threshold = int(n_emails * 0.75)

    # For rows where user_id >= top_user_threshold AND email_id >= top_email_threshold,
    # remap email_id to bottom-quartile emails and set actioned_24h = 0.
    def _force_sparse(row_email_id: int, row_user_id: int, row_actioned: int) -> tuple[int, int]:
        if row_user_id >= top_user_threshold and row_email_id >= top_email_threshold:
            # Remap to bottom-quartile email and suppress outcome
            return (row_email_id % top_email_threshold), 0
        return row_email_id, row_actioned

    email_ids = base["email_id"].to_numpy().copy()
    actioned = base["actioned_24h"].to_numpy().copy()
    user_ids = base["user_id"].to_numpy()

    for i in range(len(email_ids)):
        email_ids[i], actioned[i] = _force_sparse(
            int(email_ids[i]), int(user_ids[i]), int(actioned[i])
        )

    return base.with_columns(
        pl.Series("email_id", email_ids, dtype=pl.Int32),
        pl.Series("actioned_24h", actioned, dtype=pl.Int8),
    )


FEATURE_COLS = _FEATURE_COLS
