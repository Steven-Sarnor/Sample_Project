import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error


def generate_sample_csv(path: str, n: int = 300, random_state: int = 42):
	rng = np.random.RandomState(random_state)
	# area in sqft
	area = rng.randint(500, 4000, size=n)
	# number of rooms
	rooms = rng.randint(1, 7, size=n)
	# categorical location
	locations = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])

	# base price per sqft and location multiplier
	base_pps = 150  # base price per sqft
	loc_multiplier = {"A": 1.4, "B": 1.1, "C": 0.9}

	price = area * base_pps * np.vectorize(loc_multiplier.get)(locations)
	# rooms effect
	price = price + rooms * 10000
	# add noise
	price = price + rng.normal(0, 30000, size=n)

	# Round prices to 2 decimal places for better readability
	price = np.round(price, 2)
	
	df = pd.DataFrame({
		"area": area,
		"rooms": rooms,
		"location": locations,
		"price": price,
	})

	df.to_csv(path, index=False)
	print(f"Generated sample dataset and saved to {path}")
	return df


def load_dataset(path: str = "house_data.csv") -> pd.DataFrame:
	p = Path(path)
	if not p.exists():
		print(f"Dataset {path} not found — generating a synthetic sample dataset...")
		return generate_sample_csv(path)

	df = pd.read_csv(path)
	print(f"Loaded dataset {path} with shape {df.shape}")
	return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	# Basic inspection
	print("\nDataset head:")
	print(df.head())
	print("\nDataset description:")
	print(df.describe(include='all'))

	# Missing values handling: numeric -> median, categorical -> mode
	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	cat_cols = df.select_dtypes(include=[object, "category"]).columns.tolist()

	for c in numeric_cols:
		if df[c].isnull().any():
			med = df[c].median()
			df[c] = df[c].fillna(med)
			print(f"Filled missing numeric {c} with median={med}")

	for c in cat_cols:
		if df[c].isnull().any():
			mode = df[c].mode().iloc[0]
			df[c] = df[c].fillna(mode)
			print(f"Filled missing categorical {c} with mode={mode}")

	# Encode categorical variables (location) with one-hot encoding (drop first for interpretability)
	if "location" in df.columns:
		df = pd.get_dummies(df, columns=["location"], drop_first=True)

	return df


def train_and_evaluate(df: pd.DataFrame, target_col: str = "price"):
	# Prepare features and target
	X = df.drop(columns=[target_col])
	y = df[target_col].values

	# Ensure numeric dtype
	X = X.astype(float)

	# Split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Simple Linear Regression using area only (if present)
	if "area" in X.columns:
		X_area = X[["area"]]
		Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_area, y, test_size=0.2, random_state=42)
		lr_area = LinearRegression()
		lr_area.fit(Xa_train, ya_train)
		ya_pred = lr_area.predict(Xa_test)
		print("\nSimple Linear Regression (area only)")
		print(f"Intercept: {lr_area.intercept_:.2f}")
		print(f"Coefficient (price per sqft): {lr_area.coef_[0]:.2f}")
		print(f"R^2 (area only) on test: {r2_score(ya_test, ya_pred):.4f}")

	# Multiple Linear Regression
	lr = LinearRegression()
	lr.fit(X_train, y_train)
	y_pred = lr.predict(X_test)

	print("\nMultiple Linear Regression (all features)")
	print(f"Intercept: ₹{lr.intercept_:,.2f}")

	coef_df = pd.DataFrame({"feature": X.columns, "coefficient": lr.coef_})
	print("\nModel Coefficients:")
	print("-" * 50)
	for idx, row in coef_df.iterrows():
		print(f"{row['feature']}: ₹{row['coefficient']:,.2f}")

	r2 = r2_score(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	print(f"R^2 on test: {r2:.4f}")
	print(f"MSE on test: {mse:.2f}")

	# Residuals
	residuals = y_test - y_pred

	# Plots: predicted vs actual
	sns.set(style="whitegrid")
	plt.figure(figsize=(10, 8))
	plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions')
	plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
	plt.xlabel("Actual Price (₹)")
	plt.ylabel("Predicted Price (₹)")
	plt.title("Model Performance: Predicted vs Actual House Prices")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()
	plt.savefig("predicted_vs_actual.png", dpi=300, bbox_inches='tight')
	print("Saved plot: predicted_vs_actual.png")

	# Residuals histogram
	plt.figure(figsize=(8, 5))
	sns.histplot(residuals, kde=True)
	plt.title("Residuals distribution")
	plt.xlabel("Residual (actual - predicted)")
	plt.tight_layout()
	plt.savefig("residuals_hist.png")
	print("Saved plot: residuals_hist.png")

	# Feature importance (by absolute coefficient)
	feat_imp = coef_df.copy()
	feat_imp["abs_coef"] = feat_imp["coefficient"].abs()
	feat_imp = feat_imp.sort_values("abs_coef", ascending=False)

	plt.figure(figsize=(8, 5))
	sns.barplot(x="abs_coef", y="feature", data=feat_imp, palette="viridis")
	plt.xlabel("|Coefficient| (importance)")
	plt.title("Feature importance (by coefficient magnitude)")
	plt.tight_layout()
	plt.savefig("feature_importance.png")
	print("Saved plot: feature_importance.png")

	# Stretch goals: Ridge and Lasso
	ridge = Ridge(alpha=1.0)
	lasso = Lasso(alpha=1.0, max_iter=10000)
	ridge.fit(X_train, y_train)
	lasso.fit(X_train, y_train)
	y_ridge = ridge.predict(X_test)
	y_lasso = lasso.predict(X_test)

	print("\nRegularized models (alpha=1.0)")
	print(f"Ridge R^2 on test: {r2_score(y_test, y_ridge):.4f}")
	print(f"Lasso R^2 on test: {r2_score(y_test, y_lasso):.4f}")

	# Interpretations
	print("\nInterpretation: ")
	if "area" in X.columns:
		coef_area = coef_df.loc[coef_df.feature == "area", "coefficient"].values
		if coef_area.size:
			print(f" - Holding other features constant, price changes by {coef_area[0]:.2f} per additional sqft (coefficient of area).")
	if "rooms" in X.columns:
		coef_rooms = coef_df.loc[coef_df.feature == "rooms", "coefficient"].values
		if coef_rooms.size:
			print(f" - Each additional room changes price by {coef_rooms[0]:.2f} on average, holding others constant.")

	# Location effects: any one-hot encoded columns beginning with location_
	loc_cols = [c for c in X.columns if c.startswith("location_")]
	if loc_cols:
		print(" - Location effects (relative to dropped location):")
		for c in loc_cols:
			val = coef_df.loc[coef_df.feature == c, "coefficient"].values[0]
			print(f"    {c}: coefficient = {val:.2f}")

	# Return objects for potential further use
	return {
		"linear_model": lr,
		"ridge": ridge,
		"lasso": lasso,
		"r2": r2,
		"mse": mse,
		"coef_df": coef_df,
	}


def get_float_input(prompt, min_val=None, max_val=None):
    while True:
        try:
            val = float(input(prompt))
            if min_val is not None and val < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter a valid number.")

def predict_house_price(model, location_labels):
    while True:
        print("\n" + "="*85)
        print("🏘️  HOUSE PRICE PREDICTION - INPUT PARAMETERS 📝")
        print("="*85)
        
        print("\n📋 Please provide the following details about the house:")
        print("-" * 50)
        
        # Get house area
        print("\n📐 House Area Guidelines:")
        print("➖ Minimum: 500 square feet")
        print("➖ Maximum: 4000 square feet")
        area = get_float_input("\n📏 Enter house area (sqft): ", 500, 4000)
        
        # Get number of rooms
        print("\n🚪 Number of Rooms Guidelines:")
        print("➖ Minimum: 1 room")
        print("➖ Maximum: 7 rooms")
        rooms = get_float_input("\n🔢 Enter number of rooms: ", 1, 7)
        
        # Get location
        print("\n📍 Location Options:")
        print("-" * 20)
        print("🏢 A - Durg")
        print("🌆 B - Naya Raipur")
        print("🏙️ C - Bhilai")
        print("-" * 20)
        location = input("\nEnter location code (A, B, or C): ").upper()
        while location not in ['A', 'B', 'C']:
            print("\nInvalid location code!")
            print("Please enter:")
            print("A for Durg")
            print("B for Naya Raipur")
            print("C for Bhilai")
            location = input("\nEnter location code (A, B, or C): ").upper()
        
        # Create feature vector
        features = {
            'area': [area],
            'rooms': [rooms]
        }
        
        # Add one-hot encoded location
        for loc in location_labels[1:]:  # Skip first location (reference)
            features[f'location_{loc}'] = [1 if location == loc else 0]
        
        # Convert to DataFrame
        user_features = pd.DataFrame(features)
        
        # Get prediction
        pred = model.predict(user_features)[0]
        
        # Display results
        print("\n" + "="*85)
        print("PREDICTION RESULTS")
        print("="*85)
        
        print(f"\n📍 Property Details:")
        print("-" * 50)
        print(f"🏠 Area: {area:,.0f} square feet")
        print(f"🚪 Rooms: {rooms:,.0f}")
        city_name = ['Durg', 'Naya Raipur', 'Bhilai'][ord(location) - ord('A')]
        print(f"📌 Location: {city_name} (Zone {location})")
        
        print(f"\n💰 Estimated House Price:")
        print("-" * 50)
        print(f"₹{pred:,.2f}")
        
        # Add price range estimate (±5% variation)
        lower_estimate = pred * 0.95
        upper_estimate = pred * 1.05
        print(f"\nPrice Range (±5%):")
        print(f"Minimum: ₹{lower_estimate:,.2f}")
        print(f"Maximum: ₹{upper_estimate:,.2f}")
        
        # Ask if user wants another prediction
        print("\n" + "="*85)
        choice = input("\n🔄 Would you like to make another prediction? (y/n): ").lower()
        if choice != 'y':
            print("\n✨ Thank you for using the House Price Prediction System! ✨")
            print("👋 Goodbye!")
            break

def main():
    print("\n" + "="*85)
    print("🏠 HOUSE PRICE PREDICTION SYSTEM 💰")
    print("="*85)
    
    csv_path = "house_data.csv"
    print("\n📊 Loading and preprocessing data...")
    df = load_dataset(csv_path)
    original_df = df.copy()  # Keep a copy of original data before preprocessing
    df = preprocess(df)
    
    print("\n🤖 Training machine learning models...")
    results = train_and_evaluate(df)
    
    # Get unique location labels from original data
    location_labels = sorted(original_df['location'].unique())
    
    # Start prediction loop
    print("\n✅ Model training completed successfully!")
    print("\n🚀 Starting house price prediction system...")
    predict_house_price(results['linear_model'], location_labels)

if __name__ == "__main__":
    main()