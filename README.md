# Will this case still be pending in two years?

Code for my ES 678 project: predicting, on the day a case is filed in an Indian district court, whether it will still be undecided two years later. Running it reproduces every number, table and figure in the paper.

In short, I train on cases filed in 2013–14 and test on cases filed in 2017. That's what someone building this tool on 1 January 2017 could actually have done. The main state is West Bengal, and the same steps are repeated for Delhi, Assam and Uttarakhand.

## What you need

- Python 3.12
- About 6 GB of free disk space: 5.1 GB for the download, plus about 1 GB of extracted files
- About 8 GB of free RAM (labelling the bigger states is the hungriest part)
- About 40 minutes, most of it the first two scripts

Install the packages (the versions are pinned so the numbers come out the same):

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

The data come from the Development Data Lab's judicial data: https://www.devdatalab.org/judicial-data. That page has a Dropbox link to the whole data folder. Download it as a single zip (about 5.1 GB) and save it as:

```
data/raw/ddl_justice.zip
```

**Don't unzip it.** The code reads straight from the zip. From the command line:

```
mkdir -p data/raw
curl -L -o data/raw/ddl_justice.zip "https://www.dropbox.com/sh/hkcde3z2l1h9mq1/AADRe-BuBQ92ozAJiG7YERdCa?dl=1"
```


## Running it

Run the four scripts in order. Each one prints what it's doing and exactly which data it uses (states, filing years, number of cases and share stuck), so the log reads top to bottom.

```
python prepare_data.py          # ~10 min
python build_dataset.py         # ~10 min
python run_models.py            # ~15 min
python followup_experiment.py   # ~1 min
```

1. **`prepare_data.py`** reads the zip. It counts cases in every state, then screens states using the 2014 filings: 100k–300k filings a year, and at least 5% of cases still open. Fewer open cases suggests they're missing from the data. It keeps the case rows of the states that pass.
2. **`build_dataset.py`** works out which cases were stuck:
   - **Status:** each case is decided (has a decision date), open (no decision date, outcome recorded as missing) or unclear (dropped).
   - **Download dates:** each court was downloaded at a different time, so its date is estimated from the latest hearings of its open cases.
   - **Labels:** a case only gets a label if it was watched for two full years (plus a 60-day safety margin) before that date.
   - **Choosing states:** it applies the checks used to choose states, which were all fixed before any model was fitted, and builds the features for the four states that pass.
   - **Data facts:** it prints the facts quoted in the paper's data section, such as the missing quick cases in early years, the law-filed-under field that was filled in later, and the bail record-keeping change.
3. **`run_models.py`** does all the modelling:
   - **West Bengal:** chooses settings by training on 2013 and checking on 2016, re-checks them on a second year (train 2012, check 2015), then scores 2017 once.
   - **The other three states:** runs the same procedure.
   - **Comparisons:** scores the same model on a shuffled split, and tries the two extra ideas that didn't pan out.
   - **Model families:** the lookup table, logistic regression, gradient boosting, a single tree and a random forest, plus the two "leak" demonstrations.
4. **`followup_experiment.py`** is the graduate part. It pretends the data had been downloaded earlier, relabels the training cases three ways, and checks the paper's formula against what actually happens.

`common.py` holds the bits the scripts share: features, models and scores.

## Where things end up

- **`figures/`**: the three figures in the paper
  - `figure1_data.png`: how quickly cases get decided, pendency over time, case types
  - `figure2_four_states.png`: honest vs shuffled testing, and predicted vs actual, in four states
  - `figure3_followup.png`: the follow-up experiment
- **`results/`**: every table as a CSV. `table1.csv` is Table 1 of the paper.
- **`data/`**: the download and the intermediate files (not in the repository)

## Where each paper number comes from

| In the paper | Printed by | Saved as |
|---|---|---|
| 81 million cases, the screening of states | `prepare_data.py` | `results/cases_per_state_and_year.csv` |
| Checks used to choose states, download dates, unclear status, the three data traps, Figure 1 | `build_dataset.py` | `results/state_selection.csv`, `figures/figure1_data.png` |
| Table 1, the second check year, 7 in 10 flagged first, the leaks, the discarded models | `run_models.py` | `results/table1.csv`, `results/main_state_*.csv` |
| Four states and shuffled testing (Figure 2) | `run_models.py` | `results/four_states_summary.csv`, `figures/figure2_four_states.png` |
| Court speed and first-year shift ("did not help reliably") | `run_models.py` | `results/idea1_court_speed.csv`, `results/idea2_first_year_shift.csv` |
| Follow-up experiment and Figure 3 | `followup_experiment.py` | `results/followup_experiment.csv`, `figures/figure3_followup.png` |

Everything is deterministic (random seed 678), so a rerun gives the same numbers.

## AI Usage Disclosure

Generative AI was used as a coding assistant throughout this project. Most of the code, particularly data extraction, preprocessing, cleaning, and implementation of ML pipelines, was written with AI assistance. AI was also used for debugging, understanding unfamiliar libraries and techniques, and organizing and refining the repository to make the code cleaner, more maintainable, and reproducible. The project idea, research questions, experimental direction, and interpretation of results were my own. I reviewed the generated code manually wherever possible and used a separate AI session for additional code verification. I guided the experiments, evaluated the outputs, and made the final methodological decisions and conclusions.
