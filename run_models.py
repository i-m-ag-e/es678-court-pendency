"""Step 3 of 4: all the models in the paper (about 15 minutes).

Imagine building the tool on 1 January 2017: two-year outcomes are then known only for cases
filed up to 2014. So for the main state (West Bengal):
  * settings are chosen by training on 2013 and checking on 2016 (and re-checked by training
    on 2012 and checking on 2015);
  * the chosen settings are refitted on 2013-14 and scored once on 2017.
The same procedure is repeated for the other states that passed the checks, the same model
is also scored on a shuffled split, and two extra ideas are tried. Draws Figure 2.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (BOOSTING_GRID, FEATURES, FIGURES, LAW_FIELDS, LOOKUP_SMOOTHING, LR_PENALTIES, RESULTS, SEED,
                    CATEGORICAL, boosting_best_rounds, court_bootstrap, decision_tree, gradient_boosting, load_json,
                    load_state, logistic_regression, lookup_table, random_forest, save_json, score_table, scores,
                    show)

COLUMNS = (["year", "filed", "decided_on", "status", "labelled", "y", "court_recent_speed", "purpose"]
           + FEATURES + LAW_FIELDS)


def describe(name, part):
    return f"{name}: {len(part):,} labelled cases filed in {'+'.join(map(str, sorted(part['year'].unique())))}, " \
           f"{100 * part['y'].mean():.1f}% stuck"


def tune(train, check, with_trees=False):
    """Try every model on the checking year; each family keeps its setting with the lowest log loss."""
    rows, best = [scores(check["y"], np.full(len(check), train["y"].mean()), "same chance for everyone")], {}

    def keep(family, setting, row):
        rows.append(row)
        if family not in best or row["log_loss"] < best[family]["log_loss"]:
            best[family] = {"setting": setting, "log_loss": row["log_loss"], "row": row}

    for m in LOOKUP_SMOOTHING:
        keep("lookup", m, scores(check["y"], lookup_table(train, check, m), f"lookup table, smoothing {m}"))
    for C in LR_PENALTIES:
        p = logistic_regression(C).fit(train[FEATURES], train["y"]).predict_proba(check[FEATURES])[:, 1]
        keep("logistic regression", C, scores(check["y"], p, f"logistic regression, C={C}"))
    for params in BOOSTING_GRID:
        rounds, p = boosting_best_rounds(params, train, check)
        keep("gradient boosting", {**params, "max_iter": rounds},
             scores(check["y"], p, f"gradient boosting, step {params['learning_rate']}, "
                                   f"{params['max_leaf_nodes']} leaves, best after {rounds} rounds"))
    if with_trees:
        for depth in [3, 6, 10, 15]:
            p = decision_tree(depth).fit(train[FEATURES], train["y"]).predict_proba(check[FEATURES])[:, 1]
            keep("single tree", depth, scores(check["y"], p, f"single tree, depth {depth}"))
        for leaf in [50, 500]:
            p = random_forest(leaf).fit(train[FEATURES], train["y"]).predict_proba(check[FEATURES])[:, 1]
            keep("random forest", leaf, scores(check["y"], p, f"random forest, 100 trees, min leaf {leaf}"))
    chosen = min([f for f in best if f != "lookup"], key=lambda f: best[f]["log_loss"])
    return score_table(rows), best, chosen


def test_once(train, test, best):
    predictions = {
        "same chance for everyone": np.full(len(test), train["y"].mean()),
        "lookup table": lookup_table(train, test, best["lookup"]["setting"]),
        "logistic regression": logistic_regression(best["logistic regression"]["setting"])
        .fit(train[FEATURES], train["y"]).predict_proba(test[FEATURES])[:, 1],
        "gradient boosting": gradient_boosting(**best["gradient boosting"]["setting"])
        .fit(train[FEATURES], train["y"]).predict_proba(test[FEATURES])[:, 1],
    }
    table = score_table([scores(test["y"], p, name) for name, p in predictions.items()])
    boot = court_bootstrap(test["y"].values, predictions, test["court"].values)
    lr = boot[boot["model"] == "logistic regression"].set_index("resample")
    lk = boot[boot["model"] == "lookup table"].set_index("resample")
    diff = pd.DataFrame({m: {"difference": table.set_index("model").loc["logistic regression", m]
                                           - table.set_index("model").loc["lookup table", m],
                             "95% range": f"{(lr[m] - lk[m]).quantile(0.025):+.4f} to {(lr[m] - lk[m]).quantile(0.975):+.4f}"}
                         for m in ["brier", "pr_auc"]}).T.reset_index().rename(columns={"index": "score"})
    return table, diff, predictions


def shuffled_vs_honest(labelled, C, honest_test_prediction, test):
    pool = labelled[labelled["year"].between(2013, 2017)]
    train, check = train_test_split(pool, test_size=0.2, random_state=SEED, stratify=pool["y"])
    p = logistic_regression(C).fit(train[FEATURES], train["y"]).predict_proba(check[FEATURES])[:, 1]
    honest = scores(test["y"], honest_test_prediction, "honest: train 2013-14, test 2017")
    shuffled = scores(check["y"], p, "shuffled: random 20% of 2013-17 cases")
    return score_table([honest, shuffled])


def main_state(code, name):
    d = load_state(code, COLUMNS)
    lab = d[d["labelled"] & (d["year"] >= 2012)]
    train13, check16, train1314, test17 = (lab[lab["year"] == 2013], lab[lab["year"] == 2016],
                                           lab[lab["year"].isin([2013, 2014])], lab[lab["year"] == 2017])
    show(f"{name}: data used")
    for label, part in [("choosing settings, train", train13), ("choosing settings, check", check16),
                        ("final model, train", train1314), ("final test", test17)]:
        print("  " + describe(label, part))

    table16, best, chosen = tune(train13, check16, with_trees=True)
    C = best["logistic regression"]["setting"]
    leaks = []
    for label, cols in [("+ purpose of latest hearing", CATEGORICAL + ["purpose"]), ("+ law filed under", CATEGORICAL + LAW_FIELDS)]:
        p = logistic_regression(C, categorical=cols).fit(train13[cols + ["court_filings_last_365d"]], train13["y"]) \
            .predict_proba(check16[cols + ["court_filings_last_365d"]])[:, 1]
        leaks.append(scores(check16["y"], p, f"LEAK: logistic regression {label}"))
    table16 = pd.concat([table16, score_table(leaks)])
    table16.to_csv(RESULTS / "main_state_validation_2016.csv", index=False)
    show(f"{name}: choosing settings (train 2013, check 2016)", table16)
    show("Chosen", f"{chosen} (lowest log loss on 2016); logistic regression penalty C={C}; "
                   f"lookup smoothing {best['lookup']['setting']}; boosting {best['gradient boosting']['setting']}")

    train12, check15 = lab[lab["year"] == 2012], lab[lab["year"] == 2015]
    show("Second check year", describe("train", train12) + "\n  " + describe("check", check15))
    _, best15, chosen15 = tune(train12, check15)
    show(f"{name}: second check year (train 2012, check 2015) - best of each family",
         pd.DataFrame([{"family": f, "best setting": str(b["setting"]), "log_loss": round(b["log_loss"], 4)}
                       for f, b in best15.items()]))
    print(f"chosen on the second check year: {chosen15}")

    test_table, diff, predictions = test_once(train1314, test17, best)
    test_table.to_csv(RESULTS / "main_state_test_2017.csv", index=False)
    show(f"{name}: final test on 2017 filings (scored once)", test_table)
    show("logistic regression minus lookup table (95% ranges from resampling courts)", diff)

    families = ["lookup", "logistic regression", "gradient boosting"]
    paper_table = pd.DataFrame({
        "model": ["same chance for everyone", "lookup table", "logistic regression", "gradient boosting"],
        "check 2016 brier": [table16.iloc[0]["brier"]] + [best[f]["row"]["brier"] for f in families],
        "check 2016 pr_auc": [table16.iloc[0]["pr_auc"]] + [best[f]["row"]["pr_auc"] for f in families],
        "test 2017 brier": test_table["brier"].values,
        "test 2017 pr_auc": test_table["pr_auc"].values,
    })
    paper_table.round(3).to_csv(RESULTS / "table1.csv", index=False)
    show("Table 1 of the paper", paper_table.round(3))

    shuffle = shuffled_vs_honest(lab, C, predictions["logistic regression"], test17)
    show(f"{name}: honest vs shuffled testing (same logistic regression)", shuffle)
    save_json({"state": code, "logistic_regression_C": C}, "main_state_settings.json")
    return d, C, shuffle, test_table


def other_state(code, name):
    d = load_state(code, COLUMNS)
    lab = d[d["labelled"] & (d["year"] >= 2012)]
    train13, check16, train1314, test17 = (lab[lab["year"] == 2013], lab[lab["year"] == 2016],
                                           lab[lab["year"].isin([2013, 2014])], lab[lab["year"] == 2017])
    show(f"{name}: data used")
    for label, part in [("choosing settings, train", train13), ("choosing settings, check", check16),
                        ("final model, train", train1314), ("final test", test17)]:
        print("  " + describe(label, part))
    table16, best, chosen = tune(train13, check16)
    table16.to_csv(RESULTS / f"state_{code}_validation_2016.csv", index=False)
    print(f"chosen on 2016: {chosen}; logistic regression C={best['logistic regression']['setting']}; "
          f"lookup smoothing {best['lookup']['setting']}")
    test_table, diff, predictions = test_once(train1314, test17, best)
    test_table.to_csv(RESULTS / f"state_{code}_test_2017.csv", index=False)
    show(f"{name}: final test on 2017 filings", test_table)
    show("logistic regression minus lookup table (95% ranges from resampling courts)", diff)
    shuffle = shuffled_vs_honest(lab, best["logistic regression"]["setting"], predictions["logistic regression"], test17)
    show(f"{name}: honest vs shuffled testing", shuffle)
    return shuffle, test_table


def extra_ideas(d, name, C_main):
    """Two later ideas, both tested after the 2017 test had been seen, so 2018 Jan-Jun cases
    (never looked at before) serve as the fresh test for the first one."""
    lab = d[d["labelled"] & (d["year"] >= 2014)]
    early18 = lab[(lab["year"] == 2018) & (lab["filed"].dt.month <= 6)]
    show(f"{name}: idea 1, the court's recent speed (fresh test on Jan-Jun 2018 filings)")
    print("  " + describe("choosing settings, train", lab[lab["year"] == 2014]))
    print("  " + describe("choosing settings, check", lab[lab["year"] == 2017]))
    print("  " + describe("final model, train", lab[lab["year"].isin([2014, 2015])]))
    print("  " + describe("fresh test", early18))
    variants = {"without court speed": (), "with court speed": ("court_recent_speed",)}
    best_C, rows = {}, []
    tr, ck = lab[lab["year"] == 2014], lab[lab["year"] == 2017]
    for v, extra in variants.items():
        losses = {}
        for C in LR_PENALTIES[:-1]:
            p = logistic_regression(C, extra_numeric=extra).fit(tr[FEATURES + list(extra)], tr["y"]) \
                .predict_proba(ck[FEATURES + list(extra)])[:, 1]
            losses[C] = scores(ck["y"], p, v)["log_loss"]
        best_C[v] = min(losses, key=losses.get)
    tr = lab[lab["year"].isin([2014, 2015])]
    preds = {"lookup table": lookup_table(tr, early18, 100)}
    for v, extra in variants.items():
        preds[f"logistic regression {v}"] = logistic_regression(best_C[v], extra_numeric=extra) \
            .fit(tr[FEATURES + list(extra)], tr["y"]).predict_proba(early18[FEATURES + list(extra)])[:, 1]
    t = score_table([scores(early18["y"], p, m) for m, p in preds.items()])
    boot = court_bootstrap(early18["y"].values, preds, early18["court"].values)
    a = boot[boot["model"] == "logistic regression with court speed"].set_index("resample")["brier"]
    b = boot[boot["model"] == "logistic regression without court speed"].set_index("resample")["brier"]
    show(f"settings chosen on 2017: C = {best_C}", t)
    print(f"Brier with minus without court speed: {t.iloc[2]['brier'] - t.iloc[1]['brier']:+.4f} "
          f"(95% range {(a - b).quantile(0.025):+.4f} to {(a - b).quantile(0.975):+.4f})")
    t.to_csv(RESULTS / "idea1_court_speed.csv", index=False)

    show(f"{name}: idea 2, shift predictions by how much faster recent cases were decided in their first year")
    known = d[d["status"] != "unclear"].copy()
    known["open_after_1y"] = ~((known["status"] == "decided") & (known["decided_on"] <= known["filed"] + pd.Timedelta(days=365)))
    rows = []
    for label, train_years, recent, test_mask, C in [
            ("2016", [2013], 2014, lab["year"] == 2016, C_main),
            ("2017", [2013, 2014], 2015, lab["year"] == 2017, C_main),
            ("2018 Jan-Jun", [2014, 2015], 2016, (lab["year"] == 2018) & (lab["filed"].dt.month <= 6),
             best_C["without court speed"])]:
        tr = d[d["labelled"] & d["year"].isin(train_years)]
        te = lab[test_mask]
        q = logistic_regression(C).fit(tr[FEATURES], tr["y"]).predict_proba(te[FEATURES])[:, 1]
        old = known[known["year"].isin(train_years)]["open_after_1y"].mean()
        new = known[known["year"] == recent]["open_after_1y"].mean()
        shift = np.log(new / (1 - new)) - np.log(old / (1 - old))
        q2 = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + shift)))
        print(f"predicting {label}: open after one year {old:.3f} (training years) -> {new:.3f} ({recent}); "
              f"actually stuck {te['y'].mean():.3f}")
        rows += [{"year predicted": label, **scores(te["y"], q, "logistic regression")},
                 {"year predicted": label, **scores(te["y"], q2, "logistic regression, shifted")}]
    t2 = score_table(rows)[["year predicted", "model", "share_stuck", "avg_prediction", "brier", "pr_auc"]]
    show("idea 2 results", t2)
    t2.to_csv(RESULTS / "idea2_first_year_shift.csv", index=False)


def figure_2(rows):
    r = pd.DataFrame(rows)
    x = np.arange(len(r))
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05))
    ax = axes[0]
    ax.bar(x - 0.18, r["honest"], width=0.36, color="#d62728", label="honest: train on past, test on future")
    ax.bar(x + 0.18, r["shuffled"], width=0.36, color="#7f7f7f", label="shuffled: random 20% of all years")
    ax.axhline(0.5, color="k", lw=0.7, ls=":")
    ax.text(len(r) - 0.55, 0.505, "coin flip", fontsize=6, ha="right", va="bottom")
    ax.set_xticks(x, r["state"])
    ax.set(ylim=(0.45, 0.95), ylabel="ranking score", title="(a) Ranking: useful everywhere, flattered by shuffling")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    ax.bar(x, r["predicted"], width=0.5, color="#ff7f0e", label="model's average prediction")
    ax.scatter(x, r["actual"], marker="_", s=500, color="k", lw=2, zorder=3, label="actually stuck")
    for xi, (p, a) in enumerate(zip(r["predicted"], r["actual"])):
        ax.text(xi, max(p, a) + 0.03, "too high" if p > a + 0.03 else "too low" if p < a - 0.03 else "about right",
                ha="center", fontsize=6)
    ax.set_xticks(x, r["state"])
    ax.set(ylim=(0, 0.82), ylabel="share stuck after 2 years", title="(b) Probabilities drift with each state's trend")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIGURES / "figure2_four_states.png", dpi=300)
    plt.close(fig)
    print("saved figures/figure2_four_states.png")


def main():
    t0 = time.time()
    chosen = load_json("chosen_states.json")
    names = chosen["names"]
    fig_rows = []

    d, C, shuffle, test_table = main_state(chosen["main"], names[chosen["main"]])
    lr = test_table.set_index("model").loc["logistic regression"]
    fig_rows.append({"state": names[chosen["main"]] + "\n(main)", "honest": shuffle.iloc[0]["ranking_score"],
                     "shuffled": shuffle.iloc[1]["ranking_score"], "predicted": lr["avg_prediction"],
                     "actual": lr["share_stuck"]})
    extra_ideas(d, names[chosen["main"]], C)
    del d

    for code in chosen["others"]:
        shuffle, test_table = other_state(code, names[code])
        lr = test_table.set_index("model").loc["logistic regression"]
        fig_rows.append({"state": names[code], "honest": shuffle.iloc[0]["ranking_score"],
                         "shuffled": shuffle.iloc[1]["ranking_score"], "predicted": lr["avg_prediction"],
                         "actual": lr["share_stuck"]})
    summary = pd.DataFrame(fig_rows)
    summary.to_csv(RESULTS / "four_states_summary.csv", index=False)
    show("Figure 2 data: four states, test year 2017", summary.assign(state=summary["state"].str.replace("\n", " ")).round(3))
    figure_2(fig_rows)
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
