"""Step 4 of 4: what too-short follow-up does to the model (about 1 minute).

Pretend the main state's records had been downloaded earlier, on a date T, so that the
2013-14 training cases had only been watched for F = T - filing date days. Three ways to
label a case that was not yet decided on T:
  A. count it as stuck
  B. drop it (if it had been watched for less than two years)
  C. only use cases watched for two full years (what the paper does)
Each version trains the same logistic regression; all are scored against the TRUE outcomes
of a held-out fifth of the 2013-14 cases. The paper's formula says that under rule A the
average prediction rises by the share of training cases decided after T but within two years.
Draws Figure 3.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from common import FEATURES, FIGURES, RESULTS, SEED, TWO_YEARS, load_json, load_state, logistic_regression, scores, show

PRETEND_DOWNLOADS = ["2015-01-01", "2015-07-01", "2016-01-01", "2016-07-01", "2017-01-01"]
RULES = {"A": "not decided yet = stuck", "B": "not decided yet = dropped", "C": "only fully followed cases"}


def relabel(train, download):
    F = (pd.Timestamp(download) - train["filed"]).dt.days.values
    D = train["D"].values
    y = train["y"].values
    full = F >= TWO_YEARS
    return F, {"A": (np.ones(len(train), bool), (D > np.minimum(F, TWO_YEARS)).astype(int)),
               "B": (full | (D <= F), y),
               "C": (full, y)}


def main():
    settings = load_json("main_state_settings.json")
    code, C = settings["state"], settings["logistic_regression_C"]
    d = load_state(code, ["year", "filed", "status", "labelled", "y", "days_to_decision"] + FEATURES)
    d = d[d["labelled"] & d["year"].isin([2013, 2014])].copy()
    d["D"] = np.where(d["status"] == "decided", d["days_to_decision"], np.inf)    # never decided = infinity
    train, held_out = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d["y"])
    show("Data used", f"labelled cases filed 2013-14: {len(d):,}; training {len(train):,}, "
                      f"held-out {len(held_out):,} ({100 * held_out['y'].mean():.1f}% truly stuck)")

    truth = logistic_regression(C).fit(train[FEATURES], train["y"]).predict_proba(held_out[FEATURES])[:, 1]
    reference = scores(held_out["y"], truth, "trained on true labels")
    rows = [{**reference, "rule": "truth", "download": "-", "training_cases": len(train), "typical_followup_months": None,
             "formula_shift": 0.0, "observed_shift": 0.0, "extra_brier": 0.0, "avg_squared_shift": 0.0}]
    for download in PRETEND_DOWNLOADS:
        F, rules = relabel(train, download)
        for rule, (keep, label) in rules.items():
            if keep.sum() < 5000 or len(np.unique(label[keep])) < 2:
                print(f"download {download}, rule {rule}: only {keep.sum()} usable cases, skipped")
                continue
            q = logistic_regression(C).fit(train[FEATURES][keep], label[keep]).predict_proba(held_out[FEATURES])[:, 1]
            r = scores(held_out["y"], q, f"rule {rule}")
            rows.append({**r, "rule": rule, "download": download, "training_cases": int(keep.sum()),
                         "typical_followup_months": round(float(np.median(F)) / 30.44, 1),
                         "formula_shift": label[keep].mean() - train["y"].mean(),
                         "observed_shift": q.mean() - truth.mean(),
                         "extra_brier": r["brier"] - reference["brier"],
                         "avg_squared_shift": np.mean((q - truth) ** 2),
                         "rank_agreement_with_truth": spearmanr(q, truth).statistic})
    res = pd.DataFrame(rows).round(4)
    res.to_csv(RESULTS / "followup_experiment.csv", index=False)
    show("Results (held-out cases, true outcomes)",
         res[["download", "rule", "training_cases", "typical_followup_months", "formula_shift", "observed_shift",
              "avg_prediction", "extra_brier", "avg_squared_shift", "brier", "ranking_score", "pr_auc"]])

    # why the ranking survives: do slow-early case types stay slow at two years?
    common_types = train["case_type"].value_counts().index[:30]
    t = train[train["case_type"].isin(common_types)]
    rank_rows = []
    for days in [90, 180, 365, 545]:
        early = (t["D"] > days).groupby(t["case_type"]).mean()
        late = (t["D"] > TWO_YEARS).groupby(t["case_type"]).mean()
        gap = (early - late).sort_values(ascending=False)
        rank_rows.append({"after_days": days, "rank_agreement_across_30_case_types": round(spearmanr(early, late).statistic, 3),
                          "types_decided_in_between": ", ".join(f"{k} (+{v:.2f})" for k, v in gap.head(3).items())})
    show("Share undecided after a few months vs after two years, across the 30 commonest case types",
         pd.DataFrame(rank_rows))

    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False})
    colours = {"A": "#d62728", "B": "#1f77b4", "C": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0))
    runs = res[res["rule"] != "truth"]
    for rule, g in runs.groupby("rule"):
        g = g.sort_values("typical_followup_months")
        axes[0].plot(g["typical_followup_months"], g["avg_prediction"], marker="o", ms=3, color=colours[rule],
                     label=f"{rule}: {RULES[rule]}")
        axes[1].plot(g["typical_followup_months"], g["ranking_score"], marker="o", ms=3, color=colours[rule])
    a = runs[runs["rule"] == "A"]
    axes[0].scatter(a["typical_followup_months"], truth.mean() + a["formula_shift"], s=40, facecolors="none",
                    edgecolors="k", lw=0.7, zorder=3, label="formula's prediction for A")
    axes[0].axhline(truth.mean(), color="k", ls="--", lw=0.8, label="truth (correct labels)")
    axes[1].axhline(reference["ranking_score"], color="k", ls="--", lw=0.8)
    axes[0].set(ylim=(0, 0.8), ylabel="average predicted chance", title="(a) Wrong labels shift the probabilities")
    axes[1].set(ylabel="ranking score", title="(b) ...but rule A barely changes the ranking")
    for ax in axes:
        ax.set_xlabel("typical follow-up of training cases (months)")
    axes[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES / "figure3_followup.png", dpi=300)
    plt.close(fig)
    print("saved figures/figure3_followup.png")


if __name__ == "__main__":
    main()
