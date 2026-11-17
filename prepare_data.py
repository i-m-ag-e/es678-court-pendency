"""Step 1 of 4: take what we need out of the DDL download (about 10 minutes).

The zip is read as it is (never unzipped). This script
  1. copies the small code tables (keys) we need,
  2. looks at 2014 filings to screen states: 100k-300k filings a year, and at least 5% of
     cases still open (fewer suggests open cases are missing from the data),
  3. reads every year's case file once, counting cases per state and year for all states,
     and keeping the full rows of the screened states,
  4. keeps the law-filed-under rows (acts) of those states' cases.
Output: data/extracted/  and  results/cases_per_state_and_year.csv
"""
import io
import tarfile
import time
import zipfile
from pathlib import Path

import pandas as pd

from common import EXTRACTED, RAW_ZIP, RESULTS, save_json, show

CASES = "csv/cases/cases.tar.gz"
ACTS = "csv/acts_sections.tar.gz"
KEYS = "csv/keys/keys.tar.gz"
KEY_FILES = ["cases_state_key.csv", "type_name_key.csv", "disp_name_key.csv",
             "purpose_name_key.csv", "act_key.csv"]


class ForwardOnly(io.RawIOBase):
    """Files inside a .tar.gz can only be read front to back; pandas needs to be told."""

    def __init__(self, f):
        self.f = f

    def readable(self):
        return True

    def seekable(self):
        return False

    def readinto(self, buffer):
        data = self.f.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def csv_chunks(tar, member, chunksize=500_000):
    stream = io.BufferedReader(ForwardOnly(tar.extractfile(member)), 1 << 20)
    return pd.read_csv(stream, dtype=str, chunksize=chunksize, keep_default_na=False, na_values=[""])


def open_tar(zf, name):
    return tarfile.open(fileobj=zf.open(name), mode="r|gz")


def extract_keys(zf):
    (EXTRACTED / "keys").mkdir(exist_ok=True)
    with open_tar(zf, KEYS) as tar:
        for m in tar:
            if Path(m.name).name in KEY_FILES:
                (EXTRACTED / "keys" / Path(m.name).name).write_bytes(tar.extractfile(m).read())
    print("code tables saved:", ", ".join(KEY_FILES))


def screen_states(zf, names):
    """Screening on 2014 filings."""
    parts = []
    with open_tar(zf, CASES) as tar:
        for m in tar:
            if Path(m.name).name == "cases_2014.csv":
                for chunk in csv_chunks(tar, m):
                    parts.append(chunk.assign(open=chunk["date_of_decision"].isna())
                                 .groupby("state_code")["open"].agg(["size", "mean"]))
                break
    s = pd.concat(parts)
    s = s.assign(open_cases=s["size"] * s["mean"]).groupby(level=0)[["size", "open_cases"]].sum()
    s["state"] = s.index.map(names)
    s["pct_open"] = (100 * s["open_cases"] / s["size"]).round(1)
    s["passes"] = s["size"].between(100_000, 300_000) & (s["pct_open"] >= 5)
    s = s.sort_values("size", ascending=False).reset_index()[["state_code", "state", "size", "pct_open", "passes"]]
    show("Screening states on 2014 filings (100k-300k filings, at least 5% still open)",
         s.rename(columns={"size": "filings_2014"}))
    return {r.state_code: r.state for r in s.itertuples() if r.passes}


def extract_cases(zf, keep):
    counts = []
    with open_tar(zf, CASES) as tar:
        for m in tar:
            name = Path(m.name).name
            if not (name.startswith("cases_") and name.endswith(".csv")):
                continue
            year, t0, kept = int(name[6:10]), time.time(), 0
            for old in EXTRACTED.glob(f"cases_{year}_*.parquet"):
                old.unlink()
            for i, chunk in enumerate(csv_chunks(tar, m)):
                counts.append(chunk.assign(open=chunk["date_of_decision"].isna(), year=year)
                              .groupby(["year", "state_code"])["open"].agg(cases="size", open_cases="sum"))
                rows = chunk[chunk["state_code"].isin(keep)]
                if len(rows):
                    rows.to_parquet(EXTRACTED / f"cases_{year}_{i:03d}.parquet", index=False)
                    kept += len(rows)
            print(f"{name}: kept {kept:,} rows of the screened states ({time.time() - t0:.0f} s)")
    return pd.concat(counts).groupby(level=[0, 1]).sum().reset_index()


def extract_acts(zf, keep):
    for old in EXTRACTED.glob("acts_*.parquet"):
        old.unlink()
    kept = 0
    with open_tar(zf, ACTS) as tar:
        for m in tar:
            if m.name.endswith("acts_sections.csv"):
                for i, chunk in enumerate(csv_chunks(tar, m, 2_000_000)):
                    rows = chunk[chunk["ddl_case_id"].str[:2].isin(keep)]   # case ids start with the state code
                    rows.to_parquet(EXTRACTED / f"acts_{i:03d}.parquet", index=False)
                    kept += len(rows)
    print(f"acts_sections.csv: kept {kept:,} rows")


def main():
    t0 = time.time()
    print(f"Reading {RAW_ZIP}")
    with zipfile.ZipFile(RAW_ZIP) as zf:
        extract_keys(zf)
        names = pd.read_csv(EXTRACTED / "keys" / "cases_state_key.csv", dtype=str)
        names = names.sort_values("year").groupby("state_code")["state_name"].last().to_dict()
        keep = screen_states(zf, names)
        show("States kept for the rest of the analysis", ", ".join(keep.values()))
        save_json(keep, "screened_states.json")
        counts = extract_cases(zf, keep)
        extract_acts(zf, keep)

    counts["state"] = counts["state_code"].map(names)
    counts.to_csv(RESULTS / "cases_per_state_and_year.csv", index=False)
    show(f"All states together: {counts['cases'].sum():,} cases filed 2010-2018",
         counts.groupby("year")["cases"].sum().reset_index())
    print(f"\nDone in {(time.time() - t0) / 60:.0f} minutes.")


if __name__ == "__main__":
    main()
