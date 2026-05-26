"""
eda_and_data_gen.py
-------------------
Exploratory Data Analysis + synthetic dataset generator for Car Price Prediction.
Run this to create data/car_data.csv for training.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

np.random.seed(42)
DATA_PATH = Path("data/car_data.csv")


# ------------------------------------------------------------------
# Synthetic dataset generation (replace with real dataset if available)
# ------------------------------------------------------------------

def generate_dataset(n: int = 2000) -> pd.DataFrame:
    fuels        = ["Petrol", "Diesel", "CNG", "Electric"]
    fuel_w       = [0.50, 0.35, 0.10, 0.05]
    sellers      = ["Individual", "Dealer", "Trustmark Dealer"]
    seller_w     = [0.55, 0.35, 0.10]
    transmissions = ["Manual", "Automatic"]
    trans_w      = [0.70, 0.30]
    owners       = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"]
    owner_w      = [0.55, 0.30, 0.10, 0.05]

    years        = np.random.randint(2005, 2024, n)
    km_driven    = np.random.randint(5000, 200000, n)
    engine_cc    = np.random.choice([998, 1197, 1498, 1968, 2179, 2993], n)
    max_power    = engine_cc * np.random.uniform(0.055, 0.085, n)
    seats        = np.random.choice([5, 7, 8], n, p=[0.75, 0.20, 0.05])
    fuel         = np.random.choice(fuels, n, p=fuel_w)
    seller_type  = np.random.choice(sellers, n, p=seller_w)
    transmission = np.random.choice(transmissions, n, p=trans_w)
    owner        = np.random.choice(owners, n, p=owner_w)

    # Price formula with realistic factors
    base_price = engine_cc * 6.5
    age_factor  = 1 - 0.06 * (2024 - years)
    km_factor   = 1 - 0.0000015 * km_driven
    fuel_mult   = {"Petrol": 1.0, "Diesel": 1.15, "CNG": 0.85, "Electric": 1.40}
    trans_mult  = {"Manual": 1.0, "Automatic": 1.12}
    owner_mult  = {"First Owner": 1.0, "Second Owner": 0.88, "Third Owner": 0.76, "Fourth & Above Owner": 0.62}

    price = (
        base_price
        * np.clip(age_factor, 0.2, 1.0)
        * np.clip(km_factor,  0.3, 1.0)
        * np.array([fuel_mult[f] for f in fuel])
        * np.array([trans_mult[t] for t in transmission])
        * np.array([owner_mult[o] for o in owner])
        * np.random.uniform(0.88, 1.12, n)   # market noise
    )

    df = pd.DataFrame({
        "year": years,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "engine_cc": engine_cc,
        "max_power_bhp": np.round(max_power, 1),
        "seats": seats,
        "selling_price": np.round(price, -3),   # round to nearest 1000
    })
    return df


# ------------------------------------------------------------------
# EDA functions
# ------------------------------------------------------------------

def run_eda(df: pd.DataFrame):
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape       : {df.shape}")
    print(f"Columns     : {list(df.columns)}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive statistics:\n{df.describe().to_string()}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Car Price Prediction — EDA", fontsize=15, fontweight="bold")

    # Price distribution
    axes[0, 0].hist(df["selling_price"] / 1e5, bins=40, color="#1D9E75", edgecolor="white")
    axes[0, 0].set_title("Selling Price Distribution (Lakhs)")
    axes[0, 0].set_xlabel("Price (₹ Lakhs)")

    # Year vs Price
    axes[0, 1].scatter(df["year"], df["selling_price"] / 1e5, alpha=0.4, color="#378ADD", s=12)
    axes[0, 1].set_title("Year vs Price")
    axes[0, 1].set_xlabel("Year"); axes[0, 1].set_ylabel("Price (₹ Lakhs)")

    # KM driven vs Price
    axes[0, 2].scatter(df["km_driven"], df["selling_price"] / 1e5, alpha=0.3, color="#EF9F27", s=12)
    axes[0, 2].set_title("KM Driven vs Price")
    axes[0, 2].set_xlabel("KM Driven")

    # Fuel type
    fuel_avg = df.groupby("fuel")["selling_price"].mean().sort_values() / 1e5
    axes[1, 0].barh(fuel_avg.index, fuel_avg.values, color=["#5DCAA5", "#378ADD", "#EF9F27", "#D85A30"])
    axes[1, 0].set_title("Avg Price by Fuel Type (Lakhs)")

    # Transmission
    trans_avg = df.groupby("transmission")["selling_price"].mean() / 1e5
    axes[1, 1].bar(trans_avg.index, trans_avg.values, color=["#1D9E75", "#7F77DD"])
    axes[1, 1].set_title("Avg Price by Transmission")

    # Owner type
    owner_avg = df.groupby("owner")["selling_price"].mean().sort_values() / 1e5
    axes[1, 2].barh(owner_avg.index, owner_avg.values, color="#378ADD")
    axes[1, 2].set_title("Avg Price by Owner Type (Lakhs)")

    plt.tight_layout()
    Path("data").mkdir(exist_ok=True)
    plt.savefig("data/eda_charts.png", dpi=150, bbox_inches="tight")
    print("\nEDA charts saved to data/eda_charts.png")
    plt.close()

    # Correlation heatmap
    numeric_cols = ["year", "km_driven", "engine_cc", "max_power_bhp", "seats", "selling_price"]
    corr = df[numeric_cols].corr()
    fig2, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax, linewidths=0.5)
    ax.set_title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("data/correlation_heatmap.png", dpi=150, bbox_inches="tight")
    print("Correlation heatmap saved to data/correlation_heatmap.png")
    plt.close()


if __name__ == "__main__":
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(2000)
    df.to_csv(DATA_PATH, index=False)
    print(f"Dataset saved to {DATA_PATH} ({len(df)} rows)")
    run_eda(df)
