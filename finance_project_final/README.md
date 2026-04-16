# Expansion Readiness Project

This package contains a simplified and cleaner version of the project.

## What's different in this version
- Uses the new dataset: `data/EntireCompustat.csv`
- Uses a cleaner target variable:
  - next-year revenue growth beats either the industry median or 5%
  - next-year ROA > 2%
  - next-year CFO/Assets > 0
  - next-year Debt/Assets < 0.60
- Uses only 4 ratios inside the logistic regression model:
  - revenue_growth
  - roa
  - debt_ratio
  - cfo_assets
- Removes noisier or redundant ratios from the regression
- Tunes the probability threshold on a validation period instead of always using 0.50

## Files
- `financial_expansion_project_refined_model.ipynb` - main notebook
- `project_utils.py` - shared feature engineering and modeling helpers
- `app.py` - Streamlit app
- `artifacts/` - saved model and outputs
- `data/EntireCompustat.csv` - training dataset

## Run the notebook
Open the notebook in Colab or Jupyter and run all cells.

## Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
