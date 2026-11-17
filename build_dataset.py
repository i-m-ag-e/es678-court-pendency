"""Step 2 of 4: decide which cases were stuck, choose the states, build the features (~10 minutes).

For every screened state:
  * status: decided (has a decision date), open (no decision date, outcome recorded as missing),
    or unclear (dropped);
  * download date of each court: courts were downloaded at different times, so it is taken as
    the date by which 99% of the court's open cases had last been heard (district-wide if the
    court has fewer than 30 open cases);
  * label: stuck = no decision within two years of filing; a case gets a label only if it was
    watched for two years (plus a 60-day safety margin) before its court's download date;
  * the checks used to choose states (all fixed before any model was fitted).
For the states that pass, it saves cases + labels + features to data/processed/, prints the
facts quoted in the paper's data section, and draws Figure 1 for the main state.
"""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (EXTRACTED, FIGURES, PROCESSED, RESULTS, SAFETY_MARGIN, TWO_YEARS, build_features, clean_code,
                    clean_text, load_json, save_json, show)

COLUMNS = ["ddl_case_id", "year", "state_code", "dist_code", "court_no", "judge_position",
           "female_defendant", "female_petitioner", "female_adv_def", "female_adv_pet",
           "type_name", "purpose_name", "disp_name", "date_of_filing", "date_of_decision", "date_last_list"]
REORGANISED = {"02", "29"}            # Andhra Pradesh and Telangana split in June 2014
TRAIN_YEARS, TEST_YEAR = [2013, 2014], 2017


def to_date(s):
    d = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return d.where((d >= "2010-01-01") & (d < "2021-01-01"))      # impossible years become missing


def load_cases(code):
    parts = [pd.read_parquet(f, columns=COLUMNS, filters=[("state_code", "==", code)])
             for f in sorted(EXTRACTED.glob("cases_*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["year"] = df["year"].astype(int)
    return df


def add_labels(df):
    keys = pd.read_csv(EXTRACTED / "keys" / "disp_name_key.csv", dtype=str)
    missing_outcome = keys.loc[keys["disp_name_s"].str.strip().str.lower() == "disposition var missing"]
    missing_outcome = set(missing_outcome["year"] + "|" + clean_code(missing_outcome["disp_name"]))

    df["filed"] = to_date(df["date_of_filing"])
    df["decided_on"] = to_date(df["date_of_decision"])
    last_hearing = to_date(df["date_last_list"])
    decided = df["decided_on"].notna() & (df["decided_on"] >= df["filed"])
    still_open = (df["date_of_decision"].isna() & last_hearing.notna()
                  & (df["year"].astype(str) + "|" + clean_code(df["disp_name"])).isin(missing_outcome))
    df["status"] = np.select([decided, still_open], ["decided", "open"], "unclear")
    df.loc[df["filed"].isna(), "status"] = "unclear"

    # each court's download date
    open_cases = df[df["status"] == "open"].assign(last_hearing=last_hearing)
    court = open_cases.groupby(["dist_code", "court_no"])["last_hearing"].agg(
        court_date=lambda s: s.quantile(0.99), n="size").reset_index()
    court = court[court["n"] >= 30]
    district = open_cases.groupby("dist_code")["last_hearing"].quantile(0.99).rename("district_date").reset_index()
    df = df.merge(court[["dist_code", "court_no", "court_date"]], on=["dist_code", "court_no"], how="left")
    df = df.merge(district, on="dist_code", how="left")
    df["downloaded"] = (df["court_date"].fillna(df["district_date"])
                        .fillna(open_cases["last_hearing"].quantile(0.99)))

    deadline = df["filed"] + pd.Timedelta(days=TWO_YEARS)
    df["labelled"] = (df["status"] != "unclear") & (deadline <= df["downloaded"] - pd.Timedelta(days=SAFETY_MARGIN))
    stuck = (df["status"] == "open") | ((df["status"] == "decided") & (df["decided_on"] > deadline))
    df["y"] = np.where(df["labelled"], stuck.astype(int), -1)
    df["days_to_decision"] = (df["decided_on"] - df["filed"]).dt.days
    return df.drop(columns=["court_date", "district_date", "date_of_filing", "date_of_decision", "date_last_list"])


def selection_checks(name, code, df):
    c = df[df["year"].between(2012, TEST_YEAR)]
    per_year = c.groupby("year").size()
    jumps = (per_year / per_year.shift(1)).iloc[1:]
    known = c[c["status"] != "unclear"]
    quick = known["days_to_decision"].between(0, 90).groupby(known["year"]).mean()
    completeness = quick / quick.loc[2013:2016].median()
    test = c[(c["year"] == TEST_YEAR) & (c["status"] != "unclear")]
    all_years = df[df["year"].between(2010, TEST_YEAR)]
    row = {"state": name, "biggest_jump_in_filings": round(float(max(jumps.max(), 1 / jumps.min())), 2),
           "pct_unclear_status": round(100 * (all_years["status"] == "unclear").mean(), 2),
           "pct_2017_watched_two_years": round(100 * test["labelled"].mean(), 1),
           "quick_decisions_2013_vs_normal": round(float(completeness.get(2013, 0)), 2),
           "quick_decisions_2014_vs_normal": round(float(completeness.get(2014, 0)), 2),
           "labelled_2017_cases": int((c["year"] == TEST_YEAR).mul(c["labelled"]).sum())}
    row["no_sudden_jumps"] = bool(((jumps >= 0.5) & (jumps <= 2)).all())
    row["few_unclear"] = row["pct_unclear_status"] < 5
    row["2017_followed_long_enough"] = row["pct_2017_watched_two_years"] >= 80
    row["training_years_complete"] = bool(all(completeness.get(y, 0) >= 0.8 for y in TRAIN_YEARS))
    row["no_reorganisation"] = code not in REORGANISED
    row["passes_all"] = all(row[k] for k in ["no_sudden_jumps", "few_unclear", "2017_followed_long_enough",
                                              "training_years_complete", "no_reorganisation"])
    return row


def court_recent_speed(df):
    """Of the court's cases filed 12-24 months before this case's filing month, the share
    already decided by the start of that month (used for one idea that did not help)."""
    known = df[df["status"] != "unclear"]
    month = df["filed"].dt.to_period("M")
    out = pd.Series(np.nan, index=df.index)
    for court, g in known.groupby("court"):
        g_month, rows = g["filed"].dt.to_period("M"), df.index[df["court"] == court]
        for m in month.loc[rows].unique():
            window = (g_month >= m - 24) & (g_month < m - 12)
            if window.sum() >= 20:
                out.loc[rows[month.loc[rows] == m]] = (g["decided_on"][window] < m.start_time).mean()
    return out


def processed_state(code, df):
    types = pd.read_csv(EXTRACTED / "keys" / "type_name_key.csv", dtype=str)
    types = types.assign(year=types["year"].astype(int), type_name=clean_code(types["type_name"]))
    df = df.assign(type_name=clean_code(df["type_name"])).merge(
        types.drop_duplicates(["year", "type_name"])[["year", "type_name", "type_name_s"]],
        on=["year", "type_name"], how="left")
    purposes = pd.read_csv(EXTRACTED / "keys" / "purpose_name_key.csv", dtype=str)
    purposes = purposes.assign(year=purposes["year"].astype(int), purpose_name=clean_code(purposes["purpose_name"]))
    df = df.assign(purpose_name=clean_code(df["purpose_name"])).merge(
        purposes.drop_duplicates(["year", "purpose_name"])[["year", "purpose_name", "purpose_name_s"]],
        on=["year", "purpose_name"], how="left")
    acts = pd.concat([a[a["ddl_case_id"].str[:2] == code] for a in
                      (pd.read_parquet(f) for f in sorted(EXTRACTED.glob("acts_*.parquet")))])
    acts = acts.drop_duplicates("ddl_case_id").assign(act=lambda a: clean_code(a["act"]))
    act_names = pd.read_csv(EXTRACTED / "keys" / "act_key.csv", dtype=str)
    act_names = act_names.assign(act=clean_code(act_names["act"])).drop_duplicates("act")[["act", "act_s"]]
    df = df.merge(acts.merge(act_names, on="act", how="left"), on="ddl_case_id", how="left")
    df["has_act_row"] = df["act"].notna()

    df = df[df["year"] >= 2010].reset_index(drop=True)
    features = build_features(df)
    features["purpose"] = clean_text(df["purpose_name_s"])     # latest hearing's purpose (leak demonstration)
    out = pd.concat([df[["ddl_case_id", "year", "filed", "decided_on", "status", "labelled", "y",
                         "days_to_decision", "has_act_row", "downloaded"]], features], axis=1)
    out["court_recent_speed"] = court_recent_speed(out)
    out.to_parquet(PROCESSED / f"state_{code}.parquet", index=False)
    return out


def data_facts(name, d):
    show(f"{name}: what the data look like")
    years = d[d["year"] <= 2017]
    print(f"cases filed 2010-2017: {len(years):,}; unclear status: {100 * (years['status'] == 'unclear').mean():.1f}%")
    dl = d.drop_duplicates("court")["downloaded"]
    print(f"courts' download dates: {dl.min().date()} to {dl.max().date()} (median {dl.median().date()})")
    lab = d[d["labelled"]]
    show("share stuck after two years, by filing year (labelled cases)",
         lab.groupby("year")["y"].agg(cases="size", share_stuck="mean").round(3).reset_index())
    known = d[d["status"] != "unclear"]
    show("missing quick cases: share decided within 90 days, by filing year",
         known["days_to_decision"].between(0, 90).groupby(known["year"]).mean().round(3).reset_index()
         .rename(columns={"days_to_decision": "share_decided_within_90_days"}))
    act = lab[lab["year"].between(2012, 2017)].groupby(["year", "has_act_row"])["y"].agg(["size", "mean"]).round(3).unstack()
    act.columns = [f"{'cases' if a == 'size' else 'share stuck'}, {'with' if b else 'without'} law field"
                   for a, b in act.columns]
    show("law filed under was added later: share stuck with and without the field", act.reset_index())
    bail = lab[lab["case_type"] == "bail"]
    show("record-keeping: share of bail applications 'stuck', by filing year",
         bail.groupby("year")["y"].agg(cases="size", share_stuck="mean").round(3).reset_index())


def figure_1(d):
    plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "legend.fontsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"width_ratios": [1.05, 1.0, 1.1]})
    ax = axes[0]
    t = np.linspace(0, 8, 200)
    for year, colour in [(2010, "#d62728"), (2014, "#1f77b4"), (2017, "#ff7f0e")]:
        c = d[(d["year"] == year) & (d["status"] != "unclear")]
        days = np.sort(c["days_to_decision"].dropna().to_numpy() / 365.25)
        ax.plot(t, np.searchsorted(days, t, side="right") / len(c), color=colour, lw=1.4, label=f"filed in {year}")
    ax.axvline(2, color="k", lw=0.7, ls=":")
    ax.text(2.1, 0.02, "2 years", fontsize=6.5)
    ax.annotate("2017 stops: data\nwere collected here", xy=(3.6, 0.6), xytext=(4.0, 0.08), fontsize=6,
                arrowprops=dict(arrowstyle="->", lw=0.6))
    ax.set(xlabel="years since filing", ylabel="share of cases decided", ylim=(0, 1),
           title="(a) How quickly cases get decided")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.02))

    ax = axes[1]
    lab = d[d["labelled"]].copy()
    lab["group"] = lab["year"].astype(str)
    lab.loc[(lab["year"] == 2018) & (lab["filed"].dt.month <= 6), "group"] = "2018H1"
    order = [str(y) for y in range(2010, 2018)] + ["2018H1"]
    share = lab.groupby("group")["y"].mean().reindex(order)
    use = {"incomplete": ("2010", "2011", "2012"), "train": ("2013", "2014"), "unused": ("2015",),
           "validation": ("2016",), "test": ("2017", "2018H1")}
    colours = {"incomplete": "#7f7f7f", "train": "#1f77b4", "unused": "#cccccc", "validation": "#ff7f0e",
               "test": "#d62728"}
    colour_of = {g: colours[k] for k, gs in use.items() for g in gs}
    ax.bar(range(len(order)), share.values, color=[colour_of[g] for g in order])
    ax.set_xticks(range(len(order)), ["10", "11", "12", "13", "14", "15", "16", "17", "18*"])
    ax.tick_params(axis="x", labelsize=6)
    ax.set(xlabel="filing year (20__; 18* = Jan-Jun 2018)", ylabel="share still pending after 2 years",
           ylim=(0, 1), title="(b) Pendency falls over time")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=v, label=k) for k, v in colours.items()], frameon=False, loc="upper right",
              fontsize=6, handlelength=1, borderaxespad=0.1)

    ax = axes[2]
    names = {"bail": "bail application", "criminalmisc": "criminal misc.", "grcase": "police case (GR)",
             "titlesuit": "title (property) suit", "matrimonialsuit": "matrimonial suit", "complaintcase": "complaint case"}
    t2 = lab[lab["year"].isin([2013, 2014, 2017]) & lab["case_type"].isin(names)]
    g = t2.assign(period=np.where(t2["year"] == 2017, "2017", "2013-14")).groupby(["case_type", "period"])["y"].mean().unstack()
    g = g.reindex(g["2017"].sort_values().index)
    yy = np.arange(len(g))
    ax.barh(yy + 0.2, g["2013-14"], height=0.4, color="#1f77b4", label="filed 2013-14")
    ax.barh(yy - 0.2, g["2017"], height=0.4, color="#d62728", label="filed 2017")
    ax.set_yticks(yy, [names[c] for c in g.index])
    ax.set(xlim=(0, 1), xlabel="share still pending after 2 years", title="(c) Case type matters most")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.02, 0.0))
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIGURES / "figure1_data.png", dpi=300)
    plt.close(fig)
    print("saved figures/figure1_data.png")


def main():
    t0 = time.time()
    screened = load_json("screened_states.json")
    rows = []
    for code, name in screened.items():
        print(f"{name}: labelling ...", flush=True)
        df = add_labels(load_cases(code))
        rows.append({"code": code, **selection_checks(name, code, df)})
        del df
    checks = pd.DataFrame(rows).sort_values(["passes_all", "labelled_2017_cases"], ascending=False)
    checks.to_csv(RESULTS / "state_selection.csv", index=False)
    show("Checks used to choose states (fixed before any model was fitted)", checks.drop(columns="code"))

    passing = checks[checks["passes_all"]]
    main_state, others = passing.iloc[0]["code"], list(passing.iloc[1:]["code"])
    save_json({"main": main_state, "others": others, "names": dict(zip(checks["code"], checks["state"]))},
              "chosen_states.json")
    show("Chosen", f"main state: {screened[main_state]} (most labelled 2017 cases); "
                   f"checks repeated in: {', '.join(screened[c] for c in others)}")

    for code in [main_state] + others:
        print(f"\n{screened[code]}: building features ...", flush=True)
        d = processed_state(code, add_labels(load_cases(code)))
        print(f"saved data/processed/state_{code}.parquet: {len(d):,} cases filed 2010-2018")
        if code == main_state:
            data_facts(screened[code], d)
            figure_1(d)
        del d
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
