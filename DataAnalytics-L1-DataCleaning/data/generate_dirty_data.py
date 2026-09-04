import numpy as np
import pandas as pd
import random

np.random.seed(101)
random.seed(101)

n_rows = 1200

# 1. Customer_ID: some whitespace, some missing, duplicates
cust_ids = [f"CUST-{1000 + i%950}" for i in range(n_rows)]
# Add trailing spaces to some
cust_ids = [f" {cid} " if i % 10 == 0 else cid for i, cid in enumerate(cust_ids)]
# Introduce missing values in Customer_ID (25 rows)
for idx in np.random.choice(range(n_rows), size=25, replace=False):
    cust_ids[idx] = np.nan

# 2. Names
first_names = ["John", "Jane", "Michael", "Emily", "David", "Sarah", "James", "Emma", "Robert", "Olivia"]
last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
names = []
for i in range(n_rows):
    fn = random.choice(first_names)
    ln = random.choice(last_names)
    r = random.random()
    if r < 0.1:
        names.append(f"  {fn.lower()} {ln.lower()}  ")
    elif r < 0.2:
        names.append(f"{fn.upper()} {ln.upper()}")
    elif r < 0.23:
        names.append(np.nan)
    else:
        names.append(f"{fn} {ln}")

# 3. Gender: highly inconsistent representations
gender_variants = ["Male", "male", "M", "m", " Man ", "Female", "female", "F", "f", "Woman", "Other", np.nan]
gender_weights = [0.25, 0.15, 0.08, 0.04, 0.03, 0.23, 0.10, 0.05, 0.03, 0.02, 0.01, 0.01]
genders = random.choices(gender_variants, weights=gender_weights, k=n_rows)

# 4. Age: valid 18-75, but with missing, negative (-5), and extreme (180, 210)
ages = []
for i in range(n_rows):
    r = random.random()
    if r < 0.05:
        ages.append(np.nan)
    elif r < 0.07:
        ages.append(random.choice([-3, -12, 0])) # Impossible negative / zero
    elif r < 0.09:
        ages.append(random.choice([150, 185, 210])) # Extreme biological outliers
    elif r < 0.15:
        ages.append(f"{random.randint(22, 60)} yrs") # String formatting issue
    else:
        ages.append(random.randint(18, 75))

# 5. Join_Date: mixed date patterns, some invalid text, some nulls
dates = []
for i in range(n_rows):
    r = random.random()
    y = random.randint(2019, 2024)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    if r < 0.04:
        dates.append(np.nan)
    elif r < 0.06:
        dates.append("invalid_date_str")
    elif r < 0.35:
        dates.append(f"{y:04d}-{m:02d}-{d:02d}") # ISO
    elif r < 0.65:
        dates.append(f"{d:02d}/{m:02d}/{y:04d}") # European
    elif r < 0.85:
        dates.append(f"{m:02d}-{d:02d}-{y:04d}") # US
    else:
        dates.append(f"{pd.to_datetime(f'{y}-{m}-{d}').strftime('%B %d, %Y')}")

# 6. Annual_Income: dirty currency strings, commas, negatives, outliers, nulls
incomes = []
for i in range(n_rows):
    r = random.random()
    val = round(np.random.normal(65000, 22000), -2)
    val = max(15000, val)
    if r < 0.06:
        incomes.append(np.nan)
    elif r < 0.08:
        incomes.append("N/A")
    elif r < 0.10:
        incomes.append("-5000") # Negative income anomaly
    elif r < 0.12:
        incomes.append("$2,500,000.00") # Massive millionaire outlier
    elif r < 0.50:
        incomes.append(f"${val:,.2f}")
    elif r < 0.80:
        incomes.append(f"${val:,.0f}")
    else:
        incomes.append(f"{val}")

# 7. Credit_Score: 300 to 850 range, some out of bounds (999, 120), nulls
credit_scores = []
for i in range(n_rows):
    r = random.random()
    if r < 0.05:
        credit_scores.append(np.nan)
    elif r < 0.07:
        credit_scores.append(random.choice([999, 1050])) # Out of bounds high
    elif r < 0.08:
        credit_scores.append(random.choice([120, 150])) # Out of bounds low
    else:
        credit_scores.append(int(np.clip(np.random.normal(680, 75), 320, 840)))

# 8. Spending_Score: 1 to 100, some nulls
spending_scores = []
for i in range(n_rows):
    r = random.random()
    if r < 0.06:
        spending_scores.append(np.nan)
    elif r < 0.08:
        spending_scores.append(random.choice([145, 180])) # Outlier
    else:
        spending_scores.append(random.randint(1, 100))

# 9. City: inconsistent spelling, abbreviations, spaces
cities_raw = ["New York", "new york", " NY ", "NYC", "San Francisco", "san francisco", "SF", "Chicago", "chicago", "Chi-Town", np.nan]
city_weights = [0.25, 0.08, 0.05, 0.04, 0.22, 0.06, 0.04, 0.18, 0.04, 0.02, 0.02]
cities = random.choices(cities_raw, weights=city_weights, k=n_rows)

# 10. Loyalty_Member: Yes/No inconsistencies
loyalty_raw = ["Yes", "yes", "Y", "YES", "true", "No", "no", "N", "NO", "false", np.nan]
loyalty_weights = [0.25, 0.10, 0.08, 0.05, 0.04, 0.25, 0.10, 0.06, 0.03, 0.02, 0.02]
loyalty = random.choices(loyalty_raw, weights=loyalty_weights, k=n_rows)

df = pd.DataFrame({
    "Customer_ID": cust_ids,
    "Full_Name": names,
    "Gender": genders,
    "Age": ages,
    "Join_Date": dates,
    "Annual_Income": incomes,
    "Credit_Score": credit_scores,
    "Spending_Score": spending_scores,
    "City": cities,
    "Loyalty_Member": loyalty
})

# Inject 45 deliberate exact duplicate rows
duplicate_indices = np.random.choice(range(n_rows), size=45, replace=False)
duplicates_df = df.iloc[duplicate_indices].copy()
df = pd.concat([df, duplicates_df], ignore_index=True)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

target_csv = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L1-DataCleaning\data\dirty_customer_dataset.csv"
df.to_csv(target_csv, index=False)
print(f"Generated {len(df)} messy records to {target_csv}")
print(df.head(4))
