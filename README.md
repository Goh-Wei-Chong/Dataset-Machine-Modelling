# Dataset Machine Modelling

A two-part machine learning assignment that prepares two datasets, explores their relationships, and compares models for **employee promotion classification** and **listing price regression**. The notebooks cover missing values, feature engineering, visualisation, statistical summaries, cross-validation, and hyperparameter tuning.

## Project workflow

Run the notebooks in this order:

1. [`ML_Assignment_1.ipynb`](ML_Assignment_1.ipynb) explores the raw data and produces the prepared CSV files.
2. [`ML_Assignment_2.ipynb`](ML_Assignment_2.ipynb) loads those prepared files, creates reproducible 70/30 train/test splits, and compares six classifiers and six regressors.

| Task | Models |
| --- | --- |
| HR classification | Logistic Regression, Decision Tree, Random Forest, Multi-layer Perceptron, Support Vector Classifier, XGBoost |
| Listing price regression | Linear Regression, Decision Tree, Random Forest, Multi-layer Perceptron, Support Vector Regressor, XGBoost |

## Included datasets

All four CSV files are included. The raw inputs are preserved unchanged; the prepared files are convenient snapshots that notebook 1 can regenerate.

| File | Rows | Columns | Role and target |
| --- | ---: | ---: | --- |
| [`hr_data.csv`](hr_data.csv) | 54,808 | 14 | Raw employee data; `is_promoted` is 0 or 1. |
| [`hr_data_new.csv`](hr_data_new.csv) | 9,336 | 15 | Prepared HR data: 14 predictors and the `is_promoted` target. |
| [`listings.csv`](listings.csv) | 7,907 | 16 | Raw listing data; `price` is the regression target. |
| [`listings_new.csv`](listings_new.csv) | 6,024 | 11 | Prepared Central Region listings: 10 predictors and the `price` target. |

### HR preparation

The original HR dataset has 50,140 non-promoted and 4,668 promoted employees. Preparation retains every promoted employee and randomly samples 4,668 non-promoted employees with a fixed seed, creating a balanced dataset. It fills missing previous-year ratings with 3 and missing education values with Bachelor's, encodes categorical values, adds age bands and a training-score flag, and removes employee identifiers.

`age`, `avg_training_score`, and `length_of_service` are min-max scaled to [0, 1]. The historical feature name `good_conduct` means that the original training score was at least 80; it is not a separate conduct assessment.

Prepared columns:

```text
department, region, education, gender, recruitment_channel, no_of_trainings,
age, previous_year_rating, length_of_service, KPIs_met >80%, awards_won?,
avg_training_score, is_promoted, age_group, good_conduct
```

### Listing preparation

Preparation derives an upper price threshold of 400 from the interquartile range, retains listings at or below that threshold, and restricts the dataset to Central Region. It encodes room types and neighbourhoods, groups less frequent neighbourhoods into `Others`, fills missing monthly review rates with 0, and removes identifiers and text fields.

Review dates become elapsed days relative to the latest observed review date, **2019-08-27**. Listings without reviews receive 2,137 days before scaling, one day beyond the longest observed interval. `last_review`, `number_of_reviews`, `minimum_nights`, `calculated_host_listings_count`, and `availability_365` are min-max scaled.

**Price remains unscaled**, including zero-price rows. Regression RMSE is therefore in the source CSV's price units, MSE is in squared price units, and R² is unitless.

Prepared columns:

```text
neighbourhood, latitude, longitude, room_type, price, minimum_nights,
number_of_reviews, last_review, reviews_per_month,
calculated_host_listings_count, availability_365
```

## Setup and run

Use **Python 3.13**. On macOS, install the OpenMP runtime used by XGBoost before running the modelling notebook:

```bash
brew install libomp
```

This uses an existing Homebrew installation. See the [official XGBoost macOS dependency instructions](https://xgboost.readthedocs.io/en/stable/build.html#running-cmake-and-build).

Clone the repository and create the Python environment:

```bash
git clone https://github.com/Goh-Wei-Chong/Dataset-Machine-Modelling.git
cd Dataset-Machine-Modelling
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m notebook
```

Select the environment's Python kernel. Open `ML_Assignment_1.ipynb` and choose **Restart Kernel and Run All Cells**, then do the same for `ML_Assignment_2.ipynb`. Launch Jupyter from the repository directory so the relative CSV paths resolve correctly.

On Windows, create the environment with `py -3.13 -m venv .venv` and activate it using `.venv\Scripts\activate` in Command Prompt. For other platform-specific native requirements, consult the [XGBoost installation guide](https://xgboost.readthedocs.io/en/stable/install.html).

Install the requirements once per environment, then reinstall when `requirements.txt` changes. On subsequent visits, run these commands from the repository directory:

```bash
source .venv/bin/activate
python -m notebook
```

For an existing clone, save a copy of any local notebook changes before running `git pull`; then activate the environment and install the updated requirements. Notebook execution and autosave can modify the local `.ipynb` files.

[`requirements.txt`](requirements.txt) pins NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, statsmodels, XGBoost, and the Jupyter tools. The compatibility baseline is Python 3.13 on macOS with those pinned versions.

## Search profiles and verification

The default **standard** profile keeps every prepared row, all 12 model families, and **10-fold grid search**, using a compact grid of 2–4 candidates per model. The optional **extended** profile expands to the corrected original grids.

| Profile | Total grid candidates | Cross-validation fits in grid searches |
| --- | ---: | ---: |
| standard | 34 | 340 |
| extended | 477 | 4,770 |

These counts exclude baseline fits, final refits, and the original depth sweeps. Standalone cross-validation generally uses 5 folds; the assignment's final Decision Tree Classifier exercise retains its 200-fold evaluation. The historical grids specified **7,660 cross-validation fits** before compatibility corrections. Extended search has not been verified end to end. It retains the historical `gblinear` XGBoost booster, which emits a deprecation warning with the pinned XGBoost version.

Grid winners are reported separately from the assignment's manually specified final-model configurations. The final-model blocks preserve those configurations while using supported parameters and fixed seeds.

Run the complete preparation-and-modelling workflow from the active environment:

```bash
python scripts/verify_project.py
```

To run one stage or request the longer search:

```bash
python scripts/verify_project.py --notebook prepare
python scripts/verify_project.py --notebook models
python scripts/verify_project.py --notebook models --extended-search
```

The default command runs notebook 1 before notebook 2. Models-only verification uses the prepared CSV files already present in the repository. The verifier executes fresh kernels, validates the generated data and model outputs, saves refreshed notebook results, and writes a local report to `.verification/report.json`. The default per-cell timeout is 1,800 seconds; use `--timeout` to change it. Use `--jobs` to change the worker count from the default of 1.

For interactive notebook sessions, set `MODEL_SEARCH_PROFILE` to `standard` or `extended` before starting Jupyter. `MODEL_SEARCH_JOBS` is a positive integer controlling search workers, with a default of 1 for stable execution without process-pool warnings. Individual Random Forest and XGBoost estimators also use one worker. For example:

```bash
MODEL_SEARCH_PROFILE=standard MODEL_SEARCH_JOBS=1 python -m notebook
```

The verifier selects its profile with `--extended-search` and its worker count with `--jobs`; its defaults are standard search and 1 worker. The environment variables above apply to interactive notebook sessions.

## Results

<!-- VERIFIED_RESULTS_START -->
The complete workflow passed **224 nonempty code cells** and generated **24 figures** in a fresh Python 3.13.0 environment on macOS, using the standard profile and one search worker. Verification confirmed that the prepared CSVs reproduce the committed values, column order, and row order, and that both raw inputs remain unchanged.

The tables report held-out results from the assignment's final-model configurations. Grid-search winners appear separately in notebook 2. The HR test set contains **2,801 rows**, and the listings test set contains **1,808 rows**. Models retain the assignment's original feature subsets.

### Classification

| Final classifier | Test accuracy |
| --- | ---: |
| Logistic Regression | 74.3% |
| Decision Tree | 79.1% |
| Multi-layer Perceptron | 80.0% |
| Random Forest | 79.1% |
| Support Vector Classifier | 79.1% |
| XGBoost | 81.8% |

### Regression

| Final regressor | Test RMSE (source price units) | Test R² |
| --- | ---: | ---: |
| Linear Regression | 65.59 | 0.411 |
| Decision Tree | 59.49 | 0.516 |
| Multi-layer Perceptron | 64.60 | 0.429 |
| Random Forest | 50.81 | 0.647 |
| Support Vector Regressor | 64.36 | 0.433 |
| XGBoost | 53.81 | 0.604 |

XGBoost achieved the highest final classification accuracy in this run. Random Forest achieved the lowest final regression RMSE. The evaluation limitations below apply to both comparisons.
<!-- VERIFIED_RESULTS_END -->

## Compatibility updates

The original notebook metadata records Python 3.8.8 for preparation and 3.9.7 for modelling. The current notebooks use explicit pandas assignments, handle pandas string dtypes, convert review dates without chained assignment, and give notebook cells stable IDs. Plotting no longer applies an obsolete heatmap crop. Model parameters use supported library values, stochastic models use fixed seeds, and search failures are surfaced rather than accepted as missing scores.

## Evaluation limitations

- Preprocessing and min-max scaling are fitted before train/test splitting and cross-validation. Test data can influence these transformations, making estimates optimistic. A future evaluation should fit preprocessing inside each training fold.
- HR undersampling changes the promotion rate from about 8.5% to 50%. Accuracy on this balanced sample does not directly describe performance at the original promotion rate.
- The assignment repeatedly inspects test scores during tuning. Those scores are exploratory comparisons rather than an independent final assessment.
- Fixed integer category codes can imply ordering or distances to linear, neural-network, and support-vector models. The assignment preserves that encoding.
- The statsmodels diagnostic fits retain their original design matrices without an added intercept. Their specification therefore differs from the default scikit-learn linear and logistic fits.
- Rental findings cover the source snapshot's Central Region listings with prices from 0 to 400. They do not cover the full location or price range.

## Repository files

| File | Purpose |
| --- | --- |
| [`ML_Assignment_1.ipynb`](ML_Assignment_1.ipynb) | Exploration, cleaning, feature engineering, and prepared-data exports. |
| [`ML_Assignment_2.ipynb`](ML_Assignment_2.ipynb) | Classification, regression, statistical summaries, validation, and tuning. |
| The four CSV files above | Raw inputs and prepared snapshots required by the notebooks. |
| [`requirements.txt`](requirements.txt) | Pinned Python dependencies. |
| [`scripts/verify_project.py`](scripts/verify_project.py) | Reproducible execution and output validation. |

## Acknowledgement

The original assignment used the [scikit-learn documentation](https://scikit-learn.org/stable/) to learn model hyperparameters and grid-search configuration.
