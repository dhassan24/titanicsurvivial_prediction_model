# Titanic Survival Prediction

Binary classification model predicting Titanic passenger survival using the
[Kaggle Titanic dataset](https://www.kaggle.com/c/titanic), with feature
engineering, hyperparameter tuning, and full model evaluation.

## Pipeline

```
Raw train.csv / test.csv
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
  — Title extraction (regex) + consolidation
  — Target encoding (title → mean survival rate)
  — Binary sex encoding
  — Age imputation by title group median
  — One-hot encoding (Embarked)
        ↓
Random Forest — 5-fold Cross-Validation (baseline)
        ↓
Hyperparameter Tuning (GridSearchCV)
        ↓
Validation Set Evaluation
  — Accuracy, F1, Precision, Recall, ROC-AUC
  — Confusion matrix
  — ROC curve
        ↓
Feature Importance Analysis
        ↓
submission.csv
```

## Repository Contents

| File | Description |
|------|-------------|
| `titanic_survival_prediction.ipynb` | Full analysis notebook |
| `train.csv` | Labeled training data (891 passengers) |
| `test.csv` | Unlabeled test data (418 passengers) |
| `submission.csv` | Model predictions on test set |

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

```bash
# Clone and run
git clone https://github.com/dhassan24/titanic-survival.git
cd titanic-survival
jupyter notebook titanic_survival_prediction.ipynb
```

Update `TRAIN_PATH` and `TEST_PATH` in the configuration cell if your
data files are in a different location.

## Results

| Metric | Score |
|--------|-------|
| CV Accuracy (baseline RF) | 0.826 ± 0.022 |
| Best CV Accuracy (tuned) | GridSearchCV output |
| Validation Accuracy | See notebook §8 |
| Validation ROC-AUC | See notebook §8 |

## Feature Engineering Highlights

**Title extraction** — passenger names contain honorific titles (Mr., Mrs.,
Miss., Master., etc.) that encode both sex and social status more granularly
than the raw Sex column alone. Rare titles (Dr., Col., Rev., etc.) are
consolidated into a single `Rare` category to avoid overfitting.

**Target encoding** — titles are encoded as their mean survival rate from
the training set. This encoding is fitted on train only and applied to the
test set using the pre-fitted map, preventing data leakage.

**Age imputation by title group** — rather than imputing with the global
median (which would assign the same age to a Master and a Colonel), age is
imputed using the median within each title group, producing more realistic
estimates.

## Notes

- All deprecation warnings from the original notebook are resolved: raw
  strings for regex (`r"\w+[.]"`), `pd.set_option("future.no_silent_downcasting", True)`
  for replace(), and `fillna()` without `inplace=`.
- The cleaning pipeline is implemented as a single reusable function
  (`clean_dataset`) applied identically to train and test, eliminating
  train/test preprocessing inconsistency bugs.
- The model is refit on the full training set after validation before
  generating final test predictions — a standard practice to maximize
  the data available for the final model.
