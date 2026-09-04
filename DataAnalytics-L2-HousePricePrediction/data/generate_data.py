import numpy as np
import pandas as pd

np.random.seed(42)

n_houses = 2000

square_feet = np.random.normal(2100, 650, n_houses).astype(int)
square_feet = np.clip(square_feet, 650, 4800)

bedrooms = np.random.choice([1, 2, 3, 4, 5], size=n_houses, p=[0.05, 0.20, 0.45, 0.23, 0.07])
bathrooms = np.round(np.clip(bedrooms * 0.75 + np.random.choice([-0.5, 0.0, 0.5], size=n_houses, p=[0.2, 0.6, 0.2]), 1.0, 4.5), 1)
floors = np.random.choice([1, 2, 3], size=n_houses, p=[0.45, 0.48, 0.07])
year_built = np.random.randint(1965, 2024, size=n_houses)
house_age = 2024 - year_built

neighborhoods = np.random.choice(
    ["Suburbs", "Downtown", "Uptown", "Waterfront", "Rural"],
    size=n_houses,
    p=[0.38, 0.22, 0.20, 0.08, 0.12]
)

conditions = np.random.choice(["Fair", "Good", "Excellent"], size=n_houses, p=[0.20, 0.60, 0.20])
garage = np.random.choice([0, 1, 2, 3], size=n_houses, p=[0.10, 0.35, 0.45, 0.10])
has_pool = np.random.choice([0, 1], size=n_houses, p=[0.82, 0.18])
dist_center = np.round(np.random.exponential(10, n_houses) + 2.0, 1)
dist_center = np.clip(dist_center, 1.5, 45.0)

# Neighborhood multipliers
nb_premium = {
    "Waterfront": 160000,
    "Downtown": 95000,
    "Uptown": 75000,
    "Suburbs": 25000,
    "Rural": -15000
}

# Condition multipliers
cond_premium = {
    "Fair": -20000,
    "Good": 10000,
    "Excellent": 45000
}

base_price = 75000
noise = np.random.normal(0, 22000, n_houses)

prices = (
    base_price
    + (square_feet * 165)
    + (bedrooms * 12000)
    + (bathrooms * 18000)
    + (garage * 14000)
    + (has_pool * 32000)
    - (house_age * 950)
    - (dist_center * 2400)
    + [nb_premium[nb] for nb in neighborhoods]
    + [cond_premium[c] for c in conditions]
    + noise
)
prices = np.round(np.clip(prices, 90000, 1150000), -2)

# Introduce realistic ~2% missing values in non-critical columns
garage_col = [g if np.random.rand() > 0.02 else np.nan for g in garage]
cond_col = [c if np.random.rand() > 0.02 else np.nan for c in conditions]

df = pd.DataFrame({
    "House_ID": [f"HSE-{10001 + i}" for i in range(n_houses)],
    "Square_Feet": square_feet,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Floors": floors,
    "Year_Built": year_built,
    "House_Age": house_age,
    "Neighborhood": neighborhoods,
    "Overall_Condition": cond_col,
    "Garage_Capacity": garage_col,
    "Has_Pool": has_pool,
    "Distance_to_City_Center_km": dist_center,
    "Price": prices
})

target_csv = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L2-HousePricePrediction\data\house_prices_dataset.csv"
df.to_csv(target_csv, index=False)
print(f"Generated {len(df)} housing records to {target_csv}")
print(df.head(3))
