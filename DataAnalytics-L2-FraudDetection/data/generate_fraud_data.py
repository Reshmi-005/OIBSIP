import numpy as np
import pandas as pd

np.random.seed(42)

n_total = 5000
n_fraud = 32
n_legit = n_total - n_fraud

# Time in seconds across 48 hours (0 to 172,800)
# Legit transactions peak during daytime (9 AM to 9 PM)
legit_hours = np.random.normal(14, 4.5, n_legit) % 24
legit_time = (np.random.choice([0, 1], size=n_legit) * 86400) + (legit_hours * 3600) + np.random.uniform(0, 3600, n_legit)
legit_time = np.clip(legit_time, 0, 172800)

# Fraud transactions often cluster in early morning hours (1 AM to 5 AM) when cardholders are asleep
fraud_hours = np.random.choice([1, 2, 3, 4, 5, 23], size=n_fraud) + np.random.uniform(0, 0.99, n_fraud)
fraud_time = (np.random.choice([0, 1], size=n_fraud) * 86400) + (fraud_hours * 3600)

# Amounts
legit_amount = np.round(np.random.exponential(45, n_legit) + 1.5, 2)
legit_amount = np.clip(legit_amount, 1.0, 2500.0)

# Fraud amounts: typically either high-value unauthorized purchases or specific test transactions
fraud_amount = np.round(np.random.choice([250, 450, 780, 1200, 1.0, 2.5], size=n_fraud, p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]) + np.random.normal(0, 15, n_fraud), 2)
fraud_amount = np.clip(fraud_amount, 1.0, 3000.0)

# PCA features V1-V8
# Legit centered around 0 with std 1
legit_v = np.random.normal(0, 1.0, (n_legit, 8))

# Fraud features shifted along key diagnostic components (V1, V2, V4, V8)
fraud_v = np.random.normal(0, 1.0, (n_fraud, 8))
fraud_v[:, 0] -= 2.8 # V1 shifted negative
fraud_v[:, 1] += 2.4 # V2 shifted positive
fraud_v[:, 3] += 3.1 # V4 shifted positive
fraud_v[:, 7] -= 2.2 # V8 shifted negative

# Assemble Legit
df_legit = pd.DataFrame(legit_v, columns=[f"V{i}" for i in range(1, 9)])
df_legit['Time'] = legit_time.astype(int)
df_legit['Amount'] = legit_amount
df_legit['Class'] = 0

# Assemble Fraud
df_fraud = pd.DataFrame(fraud_v, columns=[f"V{i}" for i in range(1, 9)])
df_fraud['Time'] = fraud_time.astype(int)
df_fraud['Amount'] = fraud_amount
df_fraud['Class'] = 1

df = pd.concat([df_legit, df_fraud], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

target_csv = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L2-FraudDetection\data\fraud_detection_dataset.csv"
df.to_csv(target_csv, index=False)
print(f"Generated {len(df)} transactions to {target_csv}")
print("Class Distribution:")
print(df['Class'].value_counts())
print(f"Fraud Percentage: {df['Class'].mean()*100:.2f}%")
