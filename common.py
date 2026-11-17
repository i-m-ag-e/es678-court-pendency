"""Shared helpers: paths, labels, features, models and scores."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RAW_ZIP = ROOT / "data" / "raw" / "ddl_justice.zip"
EXTRACTED = ROOT / "data" / "extracted"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
for folder in (EXTRACTED, PROCESSED, RESULTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

SEED = 678
TWO_YEARS = 730          # days
SAFETY_MARGIN = 60       # days subtracted from a court's download date

# What the model sees (all known on the filing day)
CATEGORICAL = ["district", "court", "court_type", "case_type", "female_petitioner",
               "female_defendant", "female_adv_pet", "female_adv_def", "filing_month", "filing_weekday"]
NUMERIC = ["court_filings_last_365d"]
FEATURES = CATEGORICAL + NUMERIC
# Fields that look like filing-day information but were filled in later (used only to show the leak)
LAW_FIELDS = ["act", "criminal", "ipc_sections", "bailable"]

LR_PENALTIES = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.1, 1.0]     # scikit-learn's C: smaller = stronger
BOOSTING_GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves} for lr in [0.05, 0.2] for leaves in [15, 63]]
LOOKUP_SMOOTHING = [1, 10, 100]


# ---------------------------------------------------------------- small utilities
def show(title, table=None):
    """Print a heading and (optionally) a table, so every run log reads top to bottom."""
    print(f"\n=== {title} ===")
    if table is not None:
        print(table.to_string(index=False) if isinstance(table, pd.DataFrame) else table)


def save_json(obj, name):
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=str))


def load_json(name):
    return json.loads((RESULTS / name).read_text())


def clean_code(s):
    return s.astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def clean_text(s, keep_spaces=True):
    """Lower case, no punctuation: 'O.S.' and 'os' become the same case type."""
    s = s.fillna("").astype(str).str.lower()
    s = s.str.replace(r"\b(19|20)\d\d\b", " ", regex=True)
    s = s.str.replace(r"[^a-z0-9]+", " " if keep_spaces else "", regex=True).str.strip()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s.replace("", "missing")


# ---------------------------------------------------------------- features
def court_filings_last_365d(court, filed):
    """How many cases the same court received in the 365 days before each filing."""
    out = pd.Series(0.0, index=court.index)
    for _, idx in court.groupby(court).groups.items():
        t = filed.loc[idx].values.astype("datetime64[D]").astype(np.int64)
        order = np.argsort(t, kind="stable")
        ts = t[order]
        counts = np.empty(len(ts))
        counts[order] = np.searchsorted(ts, ts, side="left") - np.searchsorted(ts, ts - 365, side="left")
        out.loc[idx] = counts
    return out


def build_features(df):
    f = pd.DataFrame(index=df.index)
    f["district"] = df["dist_code"].astype(str)
    f["court"] = df["dist_code"].astype(str) + "_" + df["court_no"].astype(str)
    f["court_type"] = clean_text(df["judge_position"])
    f["case_type"] = clean_text(df["type_name_s"], keep_spaces=False)
    for c in ["female_petitioner", "female_defendant", "female_adv_pet", "female_adv_def"]:
        f[c] = df[c].fillna("missing").astype(str)
    f["filing_month"] = df["filed"].dt.month.astype(str)
    f["filing_weekday"] = df["filed"].dt.dayofweek.astype(str)
    f["court_filings_last_365d"] = court_filings_last_365d(f["court"], df["filed"])
    # law-filed-under fields (leak demonstration only)
    f["act"] = np.where(df["has_act_row"], clean_text(df["act_s"]), "no act row")
    f["criminal"] = df["criminal"].fillna("no act row").astype(str)
    n_ipc = pd.to_numeric(df["number_sections_ipc"], errors="coerce")
    f["ipc_sections"] = np.select([n_ipc.isna(), n_ipc >= 5], ["none", "5+"], n_ipc.fillna(0).astype(int).astype(str))
    f["bailable"] = df["bailable_ipc"].fillna("none").astype(str)
    return f


# ---------------------------------------------------------------- models
def lookup_table(train, evaluate, m):
    """Share stuck among past cases of the same type and district, pulled towards the
    overall share for small groups: (stuck + m * overall) / (cases + m)."""
    overall = train["y"].mean()
    key_train = train["case_type"] + "|" + train["district"]
    key_eval = evaluate["case_type"] + "|" + evaluate["district"]
    g = pd.DataFrame({"k": key_train.values, "y": train["y"].values}).groupby("k")["y"].agg(["sum", "size"])
    rate = (g["sum"] + m * overall) / (g["size"] + m)
    return key_eval.map(rate).fillna(overall).values


def logistic_regression(C, categorical=CATEGORICAL, extra_numeric=()):
    parts = [("categories", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), list(categorical)),
             ("workload", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), NUMERIC)]
    if extra_numeric:
        from sklearn.impute import SimpleImputer
        parts.append(("extra", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), list(extra_numeric)))
    return Pipeline([("prepare", ColumnTransformer(parts)), ("model", LogisticRegression(C=C, max_iter=2000))])


def gradient_boosting(learning_rate, max_leaf_nodes, max_iter):
    prepare = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                      min_frequency=50, max_categories=250), CATEGORICAL),
        ("workload", "passthrough", NUMERIC)])
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, max_iter=max_iter, min_samples_leaf=200,
        categorical_features=[True] * len(CATEGORICAL) + [False] * len(NUMERIC), early_stopping=False,
        random_state=SEED)
    return Pipeline([("prepare", prepare), ("model", model)])


def tree_or_forest(model):
    prepare = ColumnTransformer([
        ("categories", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50), CATEGORICAL),
        ("workload", "passthrough", NUMERIC)])
    return Pipeline([("prepare", prepare), ("model", model)])


def decision_tree(depth):
    return tree_or_forest(DecisionTreeClassifier(max_depth=depth, min_samples_leaf=200, random_state=SEED))


def random_forest(min_leaf):
    return tree_or_forest(RandomForestClassifier(n_estimators=100, min_samples_leaf=min_leaf, max_features="sqrt",
                                                 n_jobs=-1, random_state=SEED))


def boosting_best_rounds(params, train, evaluate, max_rounds=400):
    """Fit up to max_rounds and keep the round with the lowest validation log loss."""
    pipe = gradient_boosting(max_iter=max_rounds, **params).fit(train[FEATURES], train["y"])
    Z = pipe["prepare"].transform(evaluate[FEATURES])
    best_loss, best_round, best_p = np.inf, 0, None
    for i, stage in enumerate(pipe["model"].staged_predict_proba(Z), start=1):
        loss = log_loss(evaluate["y"], stage[:, 1])
        if loss < best_loss:
            best_loss, best_round, best_p = loss, i, stage[:, 1].copy()
    return best_round, best_p


# ---------------------------------------------------------------- scores
def scores(y, p, name):
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    k = int(round(0.10 * len(y)))
    top = np.argsort(-p, kind="stable")[:k]
    return {"model": name, "cases": len(y), "share_stuck": y.mean(), "avg_prediction": p.mean(),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, p),
            "pr_auc": average_precision_score(y, p), "ranking_score": roc_auc_score(y, p),
            "stuck_among_top10pct": y[top].mean()}


def score_table(rows):
    return pd.DataFrame(rows).round(4)


def court_bootstrap(y, predictions, courts, n=200):
    """95% ranges by resampling whole courts (cases in one court are alike)."""
    rng = np.random.default_rng(SEED)
    by_court = pd.Series(np.arange(len(y))).groupby(courts).apply(np.array)
    out = []
    for b in range(n):
        idx = np.concatenate([by_court.iloc[i] for i in rng.choice(len(by_court), len(by_court), replace=True)])
        for name, p in predictions.items():
            out.append({**scores(y[idx], p[idx], name), "resample": b})
    return pd.DataFrame(out)


def load_state(code, columns=None):
    return pd.read_parquet(PROCESSED / f"state_{code}.parquet", columns=columns)
