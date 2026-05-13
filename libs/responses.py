from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RunResultMetadata",
    "HarnessParallelism",
    "HarnessStageTiming",
    "HarnessPerformance",
    "HarnessDataProfile",
    "HarnessArtifact",
    "HarnessSummary",
    "HarnessResponse",
    "RunResultRow",
    "EDAFinding",
    "EDAScores",
    "EDAPerformance",
    "EDAResponse",
    "EDAResultRow",
    "FeatureGapFinding",
    "DiscoveryScores",
    "DiscoveryReproducibility",
    "DiscoveryPerformance",
    "DiscoveryResponse",
    "DiscoveryResultRow",
    "EDAConfig",
    "DiscoveryConfig",
]


# ---------------------------------------------------------------------------
# Section 1: Metadata + supporting types
# ---------------------------------------------------------------------------


class RunResultMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["run_metadata/v1"] = Field(default="run_metadata/v1", alias="schema")
    run_id: str
    project: str
    comparison_group: str
    scope: Literal["smoke", "comparison", "champion_candidate", "reproduction"]
    status: Literal[
        "completed",
        "failed_validation",
        "failed_harness",
        "failed_recording",
        "skipped",
    ] = "completed"
    verdict: Literal[
        "promote",
        "promote_conditionally",
        "repeat_with_changes",
        "discard",
        "smoke_only",
    ] = "smoke_only"
    recorded_at_utc: str
    git_sha: str = ""
    data_fingerprint: str = ""
    outcome_version: str = "v1"
    notes: str = ""


# ---------------------------------------------------------------------------
# Section 2: HarnessResponse + RunResultRow
# ---------------------------------------------------------------------------


class HarnessParallelism(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    outer_n_jobs: int = 1
    inner_threads: int = 1
    cpu_utilization_pct: float | None = None


class HarnessStageTiming(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    stage: str
    seconds: float


class HarnessPerformance(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    total_seconds: float
    parallelism: HarnessParallelism = HarnessParallelism()
    stage_timings: list[HarnessStageTiming] = []


class HarnessDataProfile(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    n_rows: int
    n_rows_source: int
    sample_frac: float
    is_full_run: bool
    n_features: int


class HarnessArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    kind: Literal["predictions", "model", "metrics", "diagnostics", "other"]
    path: str
    bytes: int = 0


class HarnessSummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    primary_metric_name: str
    primary_metric_value: float | None = None
    baseline_metric_value: float | None = None
    lift_vs_baseline: float | None = None
    secondary_metrics: dict[str, float] = {}
    direction: Literal["higher_is_better", "lower_is_better"] = "higher_is_better"


class RunResultRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["run_result_row/v1"] = Field(
        default="run_result_row/v1", alias="schema"
    )
    project: str
    run_id: str
    comparison_group: str
    scope: str
    status: str
    verdict: str
    recorded_at_utc: str
    git_sha: str
    outcome_version: str
    primary_metric_name: str
    primary_metric_value: float | None
    baseline_metric_value: float | None
    lift_vs_baseline: float | None
    metric_direction: str
    n_rows: int
    sample_frac: float
    is_full_run: bool
    n_features: int
    total_seconds: float
    cpu_utilization_pct: float | None
    fingerprint: str
    notes: str


class HarnessResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["harness/v1"] = Field(default="harness/v1", alias="schema")
    summary: HarnessSummary
    performance: HarnessPerformance
    data_profile: HarnessDataProfile
    artifacts: list[HarnessArtifact] = []
    fingerprint: str = ""

    def to_result_row(self, metadata: RunResultMetadata) -> RunResultRow:
        return RunResultRow(
            project=metadata.project,
            run_id=metadata.run_id,
            comparison_group=metadata.comparison_group,
            scope=metadata.scope,
            status=metadata.status,
            verdict=metadata.verdict,
            recorded_at_utc=metadata.recorded_at_utc,
            git_sha=metadata.git_sha,
            outcome_version=metadata.outcome_version,
            primary_metric_name=self.summary.primary_metric_name,
            primary_metric_value=self.summary.primary_metric_value,
            baseline_metric_value=self.summary.baseline_metric_value,
            lift_vs_baseline=self.summary.lift_vs_baseline,
            metric_direction=self.summary.direction,
            n_rows=self.data_profile.n_rows,
            sample_frac=self.data_profile.sample_frac,
            is_full_run=self.data_profile.is_full_run,
            n_features=self.data_profile.n_features,
            total_seconds=self.performance.total_seconds,
            cpu_utilization_pct=self.performance.parallelism.cpu_utilization_pct,
            fingerprint=self.fingerprint,
            notes=metadata.notes,
        )


# ---------------------------------------------------------------------------
# Section 3: EDAResponse + EDAResultRow
# ---------------------------------------------------------------------------


class EDAFinding(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    title: str
    severity: Literal["info", "warning", "error"] = "info"
    score: float = 0.0
    description: str = ""
    metadata: dict[str, str] = {}


class EDAScores(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    outcome_prevalence: float | None = None
    drift_score: float = 0.0
    n_findings_warning: int = 0
    n_findings_error: int = 0
    top_finding_id: str | None = None


class EDAPerformance(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    total_seconds: float
    stage_timings: list[HarnessStageTiming] = []


class EDAResultRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["eda_result_row/v1"] = Field(
        default="eda_result_row/v1", alias="schema"
    )
    project: str
    run_id: str
    comparison_group: str
    scope: str
    recorded_at_utc: str
    git_sha: str
    eda_outcome_prevalence: float | None
    eda_drift_score: float
    eda_n_findings_warning: int
    eda_n_findings_error: int
    eda_top_finding_id: str | None
    eda_total_seconds: float


class EDAResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["eda/v1"] = Field(default="eda/v1", alias="schema")
    findings: list[EDAFinding] = []
    scores: EDAScores = EDAScores()
    performance: EDAPerformance
    outcome_version: str = "v1"
    fingerprint: str = ""

    def to_eda_summary_row(self, metadata: RunResultMetadata) -> EDAResultRow:
        return EDAResultRow(
            project=metadata.project,
            run_id=metadata.run_id,
            comparison_group=metadata.comparison_group,
            scope=metadata.scope,
            recorded_at_utc=metadata.recorded_at_utc,
            git_sha=metadata.git_sha,
            eda_outcome_prevalence=self.scores.outcome_prevalence,
            eda_drift_score=self.scores.drift_score,
            eda_n_findings_warning=self.scores.n_findings_warning,
            eda_n_findings_error=self.scores.n_findings_error,
            eda_top_finding_id=self.scores.top_finding_id,
            eda_total_seconds=self.performance.total_seconds,
        )


# ---------------------------------------------------------------------------
# Section 4: DiscoveryResponse + DiscoveryResultRow + FeatureGapFinding
# ---------------------------------------------------------------------------


class FeatureGapFinding(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    id: str
    kind: Literal["residual_slice", "embedding_region", "label_noise", "drift_gap", "other"] = (
        "residual_slice"
    )
    slice_definition: str = ""
    residual_lift: float = 0.0
    severity: float = 0.0
    priority: float = 0.0
    coverage: float = 0.0
    recommended_feature_family: str = ""
    feature_plan_stub: str = ""


class DiscoveryScores(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    top_slice_residual_lift: float = 0.0
    embedding_region_count: int = 0
    label_noise_rate: float = 0.0
    priority_score_max: float = 0.0
    overall_health: Literal["green", "yellow", "red"] = "green"
    n_findings_warning: int = 0
    n_findings_error: int = 0


class DiscoveryReproducibility(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    harness_response_hash: str = ""
    predictions_fingerprint: str = ""
    data_fingerprint: str = ""


class DiscoveryPerformance(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    total_seconds: float
    stage_timings: list[HarnessStageTiming] = []


class DiscoveryResultRow(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["discovery_result_row/v1"] = Field(
        default="discovery_result_row/v1", alias="schema"
    )
    project: str
    run_id: str
    comparison_group: str
    scope: str
    recorded_at_utc: str
    git_sha: str
    discovery_top_slice_residual_lift: float
    discovery_embedding_region_count: int
    discovery_label_noise_rate: float
    discovery_priority_score_max: float
    discovery_overall_health: str
    discovery_n_findings_warning: int
    discovery_n_findings_error: int
    discovery_total_seconds: float


class DiscoveryResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    schema_version: Literal["discovery/v1"] = Field(default="discovery/v1", alias="schema")
    findings: list[FeatureGapFinding] = []
    scores: DiscoveryScores = DiscoveryScores()
    reproducibility: DiscoveryReproducibility = DiscoveryReproducibility()
    performance: DiscoveryPerformance
    outcome_version: str = "v1"

    def to_discovery_summary_row(self, metadata: RunResultMetadata) -> DiscoveryResultRow:
        return DiscoveryResultRow(
            project=metadata.project,
            run_id=metadata.run_id,
            comparison_group=metadata.comparison_group,
            scope=metadata.scope,
            recorded_at_utc=metadata.recorded_at_utc,
            git_sha=metadata.git_sha,
            discovery_top_slice_residual_lift=self.scores.top_slice_residual_lift,
            discovery_embedding_region_count=self.scores.embedding_region_count,
            discovery_label_noise_rate=self.scores.label_noise_rate,
            discovery_priority_score_max=self.scores.priority_score_max,
            discovery_overall_health=self.scores.overall_health,
            discovery_n_findings_warning=self.scores.n_findings_warning,
            discovery_n_findings_error=self.scores.n_findings_error,
            discovery_total_seconds=self.performance.total_seconds,
        )


# ---------------------------------------------------------------------------
# Section 5: Configuration types
# ---------------------------------------------------------------------------


class EDAConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled_methods: list[str] = []
    fail_on: Literal["never", "policy_error", "any_error"] = "policy_error"
    cache_dir: str | None = None


class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    enabled_methods: list[str] = []
    top_k_findings: int = 10
    cache_dir: str | None = None
    skip_when_baseline_only: bool = True
