House Price Prediction

This repository contains `ML.py`, a complete example pipeline to build a regression model that predicts house prices from features such as area (sqft), number of rooms, and location.

What the script does:
- Loads `house_data.csv` if present; otherwise generates a synthetic dataset and saves it as `house_data.csv`.
- Inspects and preprocesses the data (fills missing values, encodes categorical variables).
- Trains a Simple Linear Regression (area-only) and a Multiple Linear Regression (all features).
- Evaluates models using R² and MSE, prints coefficients and intercept.
- Saves plots: `predicted_vs_actual.png`, `residuals_hist.png`, and `feature_importance.png`.
- Trains Ridge and Lasso (alpha=1.0) and prints their R² scores.

How to run:
1. Create a Python environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Run the script:

```powershell
python .\ML.py
```

Outputs:
- Model coefficients and performance are printed to stdout.
- Plots are saved to the working directory as PNG files.

Notes:
- If you have your own `house_data.csv`, put it in the same folder. The script expects columns: `area`, `rooms`, `location`, `price`.
- To extend: add cross-validation, hyperparameter search for Ridge/Lasso, or a small Flask/FastAPI app to deploy the model.
