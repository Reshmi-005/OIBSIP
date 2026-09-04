import numpy as np
import pandas as pd

np.random.seed(42)

n_samples = 1600

# Natural wine quality distribution based on UCI Wine Quality dataset
# 3: ~1%, 4: ~3%, 5: ~42%, 6: ~40%, 7: ~12%, 8: ~2%
qualities = np.random.choice([3, 4, 5, 6, 7, 8], size=n_samples, p=[0.01, 0.03, 0.42, 0.40, 0.12, 0.02])

# Generate correlated physicochemical features
alcohol = np.random.normal(10.4, 1.0, n_samples)
# Quality correlates positively with alcohol
alcohol = alcohol + (qualities - 5.5) * 0.45
alcohol = np.clip(alcohol, 8.4, 14.5)

# Volatile acidity correlates negatively with quality (spoilage / vinegar taste)
volatile_acidity = np.random.normal(0.53, 0.17, n_samples) - (qualities - 5.5) * 0.08
volatile_acidity = np.clip(volatile_acidity, 0.12, 1.4)

# Citric acid adds freshness, correlates positively with quality
citric_acid = np.random.normal(0.27, 0.18, n_samples) + (qualities - 5.5) * 0.04
citric_acid = np.clip(citric_acid, 0.0, 0.9)

# Sulphates acts as preservative and flavor enhancer, correlates positively
sulphates = np.random.normal(0.66, 0.17, n_samples) + (qualities - 5.5) * 0.04
sulphates = np.clip(sulphates, 0.33, 1.8)

# Fixed acidity
fixed_acidity = np.clip(np.random.normal(8.3, 1.7, n_samples), 4.6, 15.0)

# Residual sugar
residual_sugar = np.clip(np.random.exponential(1.5, n_samples) + 0.9, 0.9, 14.0)

# Chlorides (saltiness)
chlorides = np.clip(np.random.normal(0.087, 0.04, n_samples) - (qualities - 5.5) * 0.005, 0.015, 0.45)

# Free & Total sulfur dioxide
free_so2 = np.clip(np.random.exponential(12, n_samples) + 2, 1, 68)
total_so2 = np.clip(free_so2 * np.random.uniform(2.2, 4.0, n_samples) + np.random.normal(10, 8, n_samples), 8, 280)

# Density
density = 0.9967 + (fixed_acidity * 0.0007) - (alcohol * 0.0008) + (residual_sugar * 0.0004)
density = np.clip(density, 0.990, 1.004)

# pH (inverse of acidity)
pH = 3.31 - (fixed_acidity - 8.3) * 0.05 + np.random.normal(0, 0.12, n_samples)
pH = np.clip(pH, 2.8, 4.0)

df = pd.DataFrame({
    "fixed_acidity": np.round(fixed_acidity, 2),
    "volatile_acidity": np.round(volatile_acidity, 3),
    "citric_acid": np.round(citric_acid, 2),
    "residual_sugar": np.round(residual_sugar, 2),
    "chlorides": np.round(chlorides, 4),
    "free_sulfur_dioxide": np.round(free_so2, 1),
    "total_sulfur_dioxide": np.round(total_so2, 1),
    "density": np.round(density, 4),
    "pH": np.round(pH, 2),
    "sulphates": np.round(sulphates, 2),
    "alcohol": np.round(alcohol, 2),
    "quality": qualities
})

target_csv = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L2-WineQualityPrediction\data\wine_quality_dataset.csv"
df.to_csv(target_csv, index=False)
print(f"Generated {len(df)} wine samples to {target_csv}")
print("Quality Distribution:")
print(df['quality'].value_counts().sort_index())
