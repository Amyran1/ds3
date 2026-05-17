from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

_USER_EMAILS_VERSION = 4
FULL_WORK_DF_CACHE_SCHEMA_VERSION = 1
_CONFOUND_CACHE_SCHEMA_VERSION = 1


def _find_project_root() -> Path:
    here = Path(__file__).parent
    repo_root = here.parent.parent
    return repo_root


def _blake2b_hex(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def _blake2b_bytes(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=8).hexdigest()


def confound_data_fingerprint(df: pl.DataFrame) -> str:
    """Fingerprint for confound score cache.

    Depends ONLY on the columns that affect confound computation (user_id,
    email_id, date_sent, actioned_24h). Does NOT include feature column names
    or dtypes — confounds are invariant to which feature_cols are selected.
    """
    date_col = df.get_column("date_sent")
    parts = [
        f"v{_CONFOUND_CACHE_SCHEMA_VERSION}",
        f"u{_USER_EMAILS_VERSION}",
        f"n{df.height}",
        f"mn{date_col.min()}",
        f"mx{date_col.max()}",
        f"os{int(df.get_column('actioned_24h').cast(pl.Int64).sum())}",
    ]
    return _blake2b_hex("|".join(parts))


def data_fingerprint(df: pl.DataFrame) -> str:
    row_count = len(df)
    date_col = df["date_sent"]
    min_date = str(date_col.min())
    max_date = str(date_col.max())
    schema_key = ",".join(sorted(df.columns))
    outcome_sum = (
        int(df["actioned_24h"].cast(pl.Int64).sum()) if "actioned_24h" in df.columns else 0
    )
    raw = (
        f"v{_USER_EMAILS_VERSION}|{row_count}|{min_date}|{max_date}|{schema_key}|acts={outcome_sum}"
    )
    return _blake2b_hex(raw)


def full_work_df_fingerprint(df: pl.DataFrame, outcome_variable: str) -> str:
    row_count = len(df)
    date_col = df["date_sent"]
    min_date = str(date_col.min())
    max_date = str(date_col.max())
    schema_key = ",".join(f"{c}:{t}" for c, t in sorted(df.schema.items()))
    outcome_sum = (
        int(df["actioned_24h"].cast(pl.Int64).sum()) if "actioned_24h" in df.columns else 0
    )
    raw = (
        f"fwdf_sv{FULL_WORK_DF_CACHE_SCHEMA_VERSION}"
        f"|uev{_USER_EMAILS_VERSION}"
        f"|{outcome_variable}"
        f"|{row_count}|{min_date}|{max_date}"
        f"|{schema_key}"
        f"|acts={outcome_sum}"
    )
    return _blake2b_hex(raw)


def fold_train_fingerprint(train_df: pl.DataFrame, fold_k: int, user_features: list[str]) -> str:
    base_fp = data_fingerprint(train_df)
    features_key = ",".join(sorted(user_features))
    raw = f"{base_fp}|fold_{fold_k}|{features_key}"
    return _blake2b_hex(raw)


class HarnessCache:
    def __init__(self, project_root: Path | None = None) -> None:
        self.root = (
            (project_root or _find_project_root())
            / "data"
            / "projects"
            / "civic_shout_action_rate_increase"
            / "harness_cache"
        )
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def base_path(self) -> Path:
        return self.root

    def _confound_path(self, fingerprint: str) -> Path:
        return self.root / f"confound_scores_{fingerprint}.parquet"

    def _ume_path(self, fingerprint: str, fold_k: int) -> Path:
        return self.root / f"user_main_effect_predictions_fold_{fold_k}_{fingerprint}.parquet"

    def get_confound_scores(self, fingerprint: str) -> pl.DataFrame | None:
        path = self._confound_path(fingerprint)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def put_confound_scores(self, fingerprint: str, df: pl.DataFrame) -> None:
        path = self._confound_path(fingerprint)
        df.write_parquet(path)

    def get_user_main_effect_predictions(
        self, fingerprint: str, fold_k: int
    ) -> pl.DataFrame | None:
        path = self._ume_path(fingerprint, fold_k)
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def put_user_main_effect_predictions(
        self, fingerprint: str, fold_k: int, df: pl.DataFrame
    ) -> None:
        path = self._ume_path(fingerprint, fold_k)
        df.write_parquet(path)

    def get_full_work_df(self, fingerprint: str) -> pl.DataFrame | None:
        """Load cached full_work_df (post-confound-join, pre-sample) for the given fingerprint.
        Returns None on cache miss."""
        path = self.root / f"full_work_df_{fingerprint}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path)

    def put_full_work_df(self, fingerprint: str, df: pl.DataFrame) -> None:
        """Persist full_work_df to cache."""
        path = self.root / f"full_work_df_{fingerprint}.parquet"
        df.write_parquet(path)
