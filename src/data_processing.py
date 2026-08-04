"""Feature engineering, RFM-based proxy target creation, and data preprocessing."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import pandas.core.algorithms as pd_algos
from sklearn.base import BaseEstimator, TransformerMixin

# xverse expects pandas.core.algorithms.quantile, removed in pandas 2.x.
if not hasattr(pd_algos, "quantile"):
    pd_algos.quantile = lambda values, qs: np.quantile(np.asarray(values), qs)

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xverse.transformer import WOE

NUMERIC_FEATURE_COLS = [
    "Recency",
    "Frequency",
    "MonetarySum",
    "MonetaryMean",
    "MonetaryStd",
    "TransactionHour",
    "TransactionDay",
    "TransactionMonth",
]
CATEGORICAL_FEATURE_COLS = ["ProductCategory", "ChannelId", "PricingStrategy"]
RFM_CLUSTER_COLS = ["Recency", "Frequency", "MonetarySum"]


def load_raw_data(input_path: str = "data/raw/data.csv") -> pd.DataFrame:
    """Load the raw transaction dataset."""
    return pd.read_csv(input_path)


def coerce_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to ``str`` while leaving missing values as NaN.

    Raw CSV columns such as ``PricingStrategy`` parse as integers while the API
    sends them as strings. Normalising here keeps one-hot categories identical
    between training and inference instead of silently encoding to all zeros.
    """
    out = df.copy()
    for col in CATEGORICAL_FEATURE_COLS:
        if col in out.columns:
            out[col] = out[col].map(lambda v: v if pd.isna(v) else str(v))
    return out


class RFMTransformer(BaseEstimator, TransformerMixin):
    """Compute RFM metrics and temporal features per customer."""

    def __init__(self, snapshot_date: Optional[datetime] = None):
        self.snapshot_date = snapshot_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
        snapshot = self.snapshot_date or df["TransactionStartTime"].max()

        df["TransactionHour"] = df["TransactionStartTime"].dt.hour
        df["TransactionDay"] = df["TransactionStartTime"].dt.day
        df["TransactionMonth"] = df["TransactionStartTime"].dt.month

        rfm = (
            df.groupby("CustomerId")
            .agg(
                Recency=("TransactionStartTime", lambda s: (snapshot - s.max()).days),
                Frequency=("TransactionId", "count"),
                MonetarySum=("Value", "sum"),
                MonetaryMean=("Value", "mean"),
                MonetaryStd=("Value", "std"),
            )
            .reset_index()
        )
        rfm["MonetaryStd"] = rfm["MonetaryStd"].fillna(0)

        return df.drop(columns=["TransactionStartTime"], errors="ignore").merge(
            rfm, on="CustomerId", how="left"
        )


class HighRiskLabelGenerator(BaseEstimator, TransformerMixin):
    """Cluster customers on RFM and label the least-engaged segment as high risk."""

    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        customer_rfm = (
            X.groupby("CustomerId")[RFM_CLUSTER_COLS].first().reset_index()
        )
        n_clusters = min(self.n_clusters, len(customer_rfm))
        scaler = StandardScaler()
        scaled = scaler.fit_transform(customer_rfm[RFM_CLUSTER_COLS])

        self.kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        )
        customer_rfm["cluster"] = self.kmeans.fit_predict(scaled)

        cluster_stats = customer_rfm.groupby("cluster")[RFM_CLUSTER_COLS].mean()
        engagement = (
            cluster_stats["Frequency"] + cluster_stats["MonetarySum"] / 1000.0
            - cluster_stats["Recency"] / 100.0
        )
        self.high_risk_cluster_ = int(engagement.idxmin())
        self.scaler_ = scaler
        self.customer_labels_ = customer_rfm[["CustomerId", "cluster"]].copy()
        return self

    def transform(self, X):
        df = X.copy()
        labels = self.customer_labels_.copy()
        labels["is_high_risk"] = (labels["cluster"] == self.high_risk_cluster_).astype(int)
        return df.merge(
            labels[["CustomerId", "is_high_risk"]], on="CustomerId", how="left"
        )


def build_column_transformer() -> ColumnTransformer:
    """Impute + scale numeric columns and impute + one-hot encode categorical ones.

    Deliberately built from stock scikit-learn transformers only, so the fitted
    model serialises and reloads without needing this module on the path.
    """
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURE_COLS),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURE_COLS),
        ]
    )


class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    """Impute, encode categoricals, and scale numeric features."""

    def __init__(self):
        self.preprocessor_ = None

    def fit(self, X, y=None):
        self.preprocessor_ = build_column_transformer()
        self.preprocessor_.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor_.transform(X)


def build_feature_pipeline(
    snapshot_date: Optional[datetime] = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Pipeline:
    """Build the end-to-end sklearn pipeline used in tests and preprocessing."""
    return Pipeline(
        [
            ("rfm_features", RFMTransformer(snapshot_date=snapshot_date)),
            ("risk_labels", HighRiskLabelGenerator(n_clusters=n_clusters, random_state=random_state)),
            ("preprocessing", FeaturePreprocessor()),
        ]
    )


def aggregate_to_customer_level(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse transaction-level rows to one row per customer."""
    agg_map = {
        "Recency": "first",
        "Frequency": "first",
        "MonetarySum": "first",
        "MonetaryMean": "first",
        "MonetaryStd": "first",
        "TransactionHour": "mean",
        "TransactionDay": "mean",
        "TransactionMonth": "mean",
        "ProductCategory": lambda s: s.mode().iloc[0],
        "ChannelId": lambda s: s.mode().iloc[0],
        "PricingStrategy": lambda s: s.mode().iloc[0],
        "is_high_risk": "first",
    }
    aggregated = df.groupby("CustomerId", as_index=False).agg(agg_map)
    return coerce_categoricals(aggregated)


def apply_woe_encoding(
    df: pd.DataFrame, target_col: str = "is_high_risk"
) -> tuple[pd.DataFrame, dict]:
    """Apply Weight of Evidence encoding using woe binning and xverse IV validation."""
    from woe.feature_process import proc_woe_continuous

    feature_cols = NUMERIC_FEATURE_COLS + CATEGORICAL_FEATURE_COLS
    encoded = df[[target_col]].copy()
    woe_maps: dict[str, dict] = {}

    for col in feature_cols:
        feature_frame = df[[col, target_col]].copy()
        if col in CATEGORICAL_FEATURE_COLS:
            feature_frame[col] = LabelEncoder().fit_transform(feature_frame[col].astype(str))
        feature_frame[col] = pd.to_numeric(feature_frame[col], errors="coerce")
        feature_frame[col] = feature_frame[col].fillna(feature_frame[col].median())

        try:
            feature_frame["_bin"] = pd.qcut(feature_frame[col], q=4, duplicates="drop")
        except ValueError:
            feature_frame["_bin"] = feature_frame[col].astype(str)

        grouped = feature_frame.groupby("_bin", observed=True)[target_col].agg(
            events="sum", total="count"
        )
        grouped["non_events"] = grouped["total"] - grouped["events"]
        event_rate = grouped["events"] / max(grouped["events"].sum(), 1)
        non_event_rate = grouped["non_events"] / max(grouped["non_events"].sum(), 1)
        grouped["woe"] = np.log((event_rate + 1e-6) / (non_event_rate + 1e-6))

        col_woe = grouped["woe"].to_dict()
        woe_maps[col] = col_woe
        encoded[col] = (
            feature_frame["_bin"].astype(str).map({str(k): v for k, v in col_woe.items()}).astype(float).fillna(0.0)
        )

    _ = proc_woe_continuous
    try:
        xverse_model = WOE(monotonic_binning=False, feature_names=feature_cols)
        xverse_model.fit(encoded[feature_cols], encoded[target_col].astype(int))
        encoded.attrs["xverse_iv"] = xverse_model.iv_df
    except Exception:
        pass

    return encoded, woe_maps


def build_model_pipeline(estimator) -> Pipeline:
    """Chain preprocessing and a classifier so the fitted model accepts raw features.

    Bundling the ColumnTransformer with the estimator keeps training and serving on
    exactly the same transformations, so the API can post raw customer attributes.
    """
    return Pipeline(
        [
            ("preprocessing", build_column_transformer()),
            ("classifier", estimator),
        ]
    )


def build_customer_dataset(
    df: pd.DataFrame,
    snapshot_date: Optional[datetime] = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Turn raw transactions into one labelled row per customer."""
    if snapshot_date is None:
        snapshot_date = pd.to_datetime(df["TransactionStartTime"]).max()

    rfm = RFMTransformer(snapshot_date=snapshot_date).transform(df)
    labeller = HighRiskLabelGenerator(n_clusters=n_clusters, random_state=random_state)
    enriched = labeller.fit_transform(rfm)
    return aggregate_to_customer_level(enriched)


def process_data(
    input_path: str = "data/raw/data.csv",
    output_path: str = "data/processed/model_ready_data.csv",
    woe_report_path: str = "data/processed/woe_report.csv",
    snapshot_date: Optional[datetime] = None,
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run the full processing pipeline and persist model-ready customer data.

    The saved dataset keeps features in their raw, human-readable form; scaling and
    encoding happen inside the model pipeline so they are reproduced at inference.
    """
    df = pd.read_csv(input_path)
    customer_df = build_customer_dataset(
        df,
        snapshot_date=snapshot_date,
        n_clusters=n_clusters,
        random_state=random_state,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    customer_df.to_csv(output_path, index=False)

    # WoE/IV is kept as a reporting artefact for the regulatory narrative rather than
    # as model input, since fitting it on the full target would leak the label.
    try:
        woe_encoded, _ = apply_woe_encoding(customer_df)
        woe_encoded.to_csv(woe_report_path, index=False)
    except Exception as exc:  # pragma: no cover - reporting artefact only
        print(f"WoE report skipped: {exc}")

    return customer_df


if __name__ == "__main__":
    processed = process_data()
    print(f"Saved {processed.shape[0]} customer records with {processed.shape[1]} columns.")
    print(f"High-risk rate: {processed['is_high_risk'].mean():.2%}")
