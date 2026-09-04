import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# Generate 4,000 transaction records spanning 2023-01-01 to 2023-12-31
n_records = 4000
start_date = datetime(2023, 1, 1)
end_date = datetime(2023, 12, 31)
days_span = (end_date - start_date).days

# Pool of 600 unique customers with different purchasing habits
# Some frequent big spenders, some one-time shoppers, some lapsed
customers = [f"CUST-{10000 + i}" for i in range(600)]

# Customer behavioral weights:
# 10% Champions (frequent, recent, high spend)
# 25% Loyal (moderate-high frequency, good spend)
# 30% Occasional / Recent (low frequency, recent)
# 35% Lapsed / Hibernating (visited early in 2023, never returned)
cust_archetypes = {}
for i, cid in enumerate(customers):
    if i < 60:
        cust_archetypes[cid] = "champion"
    elif i < 210:
        cust_archetypes[cid] = "loyal"
    elif i < 390:
        cust_archetypes[cid] = "occasional"
    else:
        cust_archetypes[cid] = "at_risk_lapsed"

# Product catalogue
catalog = [
    ("85123A", "WHITE HANGING HEART T-LIGHT HOLDER", 2.95),
    ("71053", "WHITE METAL LANTERN", 3.75),
    ("84406B", "CREAM CUPID HEARTS COAT HANGER", 4.25),
    ("84029G", "KNITTED UNION FLAG HOT WATER BOTTLE", 4.95),
    ("84029E", "RED WOOLLY HOTTIE WHITE HEART", 4.95),
    ("22752", "SET 7 BABUSHKA NESTING BOXES", 8.50),
    ("21730", "GLASS STAR FROSTED T-LIGHT HOLDER", 3.25),
    ("22632", "HAND WARMER RED RETROSPOT", 2.10),
    ("22633", "HAND WARMER UNION JACK", 2.10),
    ("22720", "SET OF 3 CAKE TINS PANTRY DESIGN", 12.75),
    ("22197", "SMALL POPCORN HOLDER", 0.85),
    ("84879", "ASSORTED COLOUR BIRD ORNAMENT", 1.69),
    ("22086", "PAPER CHAIN KIT 50'S CHRISTMAS", 2.55),
    ("22960", "JAM MAKING SET WITH JARS", 4.95),
    ("23084", "RABBIT NIGHT LIGHT", 2.08)
]

records = []
invoice_counter = 500000

for i in range(n_records):
    # Pick customer based on archetype frequency
    # Champions purchase often throughout the year
    # At-risk purchased early in the year
    r = np.random.rand()
    if r < 0.35: # Champion order
        cid = np.random.choice(customers[:60])
        day_offset = np.random.randint(200, days_span) # Late in the year
    elif r < 0.65: # Loyal order
        cid = np.random.choice(customers[60:210])
        day_offset = np.random.randint(60, days_span)
    elif r < 0.85: # Occasional
        cid = np.random.choice(customers[210:390])
        day_offset = np.random.randint(180, days_span)
    else: # Lapsed / At-risk
        cid = np.random.choice(customers[390:])
        day_offset = np.random.randint(0, 150) # Early in year only

    inv_date = start_date + timedelta(days=int(day_offset), hours=int(np.random.randint(8, 20)), minutes=int(np.random.randint(0, 59)))
    item = catalog[np.random.randint(0, len(catalog))]
    
    qty = int(np.random.choice([1, 2, 4, 6, 12, 24], p=[0.35, 0.25, 0.20, 0.10, 0.07, 0.03]))
    price = item[2]
    
    # 2% deliberate missing CustomerID to simulate real-world e-commerce data hygiene requirements
    cust_val = cid if np.random.rand() > 0.025 else np.nan
    
    # 1.5% cancellations with negative quantity and 'C' prefix in InvoiceNo
    is_cancelled = np.random.rand() < 0.015
    inv_no = f"C{invoice_counter + i//3}" if is_cancelled else f"{invoice_counter + i//3}"
    if is_cancelled:
        qty = -qty

    country = np.random.choice(["United Kingdom", "Germany", "France", "Spain", "Netherlands"], p=[0.82, 0.07, 0.05, 0.03, 0.03])

    records.append({
        "InvoiceNo": inv_no,
        "StockCode": item[0],
        "Description": item[1],
        "Quantity": qty,
        "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
        "UnitPrice": price,
        "CustomerID": cust_val,
        "Country": country
    })

df = pd.DataFrame(records)
target_path = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L1-CustomerSegmentation\data\ecommerce_customer_data.csv"
df.to_csv(target_path, index=False)
print(f"Generated {len(df)} transactions to {target_path}")
print(df.head(4))
