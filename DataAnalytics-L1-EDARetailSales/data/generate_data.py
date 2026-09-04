import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

n_records = 2500

# Dates over 2 full years (2023-01-01 to 2024-12-31)
start_date = datetime(2023, 1, 1)
end_date = datetime(2024, 12, 31)
days_range = (end_date - start_date).days

random_days = np.random.randint(0, days_range, size=n_records)
# Seasonality: higher sales in Q4 (Oct-Dec)
q4_weights = [1.5 if (start_date + timedelta(days=int(d))).month in [10, 11, 12] else 1.0 for d in random_days]
selected_indices = np.random.choice(range(n_records), size=n_records, p=np.array(q4_weights)/sum(q4_weights))
dates = [start_date + timedelta(days=int(random_days[i])) for i in selected_indices]
dates.sort()

# Customers
n_customers = 850
customer_pool = [f"CUST-{1000 + i}" for i in range(n_customers)]
cust_genders = {cid: np.random.choice(["Male", "Female"], p=[0.48, 0.52]) for cid in customer_pool}
cust_ages = {cid: int(np.clip(np.random.normal(36, 12), 18, 72)) for cid in customer_pool}

assigned_custs = np.random.choice(customer_pool, size=n_records)
genders = [cust_genders[cid] for cid in assigned_custs]
ages = [cust_ages[cid] for cid in assigned_custs]

def get_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 50:
        return "36-50"
    else:
        return "51+"

age_groups = [get_age_group(a) for a in ages]

# Categories and Products
products_catalog = {
    "Electronics": [
        ("Wireless Noise-Canceling Headphones", 149.99, 85.00),
        ("Smart Fitness Watch", 199.99, 110.00),
        ("Bluetooth Portable Speaker", 59.99, 32.00),
        ("Ultra-Slim Laptop Stand", 39.99, 18.00),
        ("4K Ultra HD Streaming Stick", 49.99, 28.00),
    ],
    "Clothing": [
        ("Organic Cotton Crew T-Shirt", 24.99, 9.50),
        ("Denim Slim Fit Jeans", 59.99, 26.00),
        ("Waterproof Hooded Windbreaker", 89.99, 42.00),
        ("Classic Wool Blend Sweater", 69.99, 31.00),
        ("Athletic Performance Joggers", 44.99, 19.00),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Cookware Set (10-Pc)", 219.99, 125.00),
        ("Compact Espresso Machine", 179.99, 98.00),
        ("Non-Stick Ceramic Frying Pan", 34.99, 15.00),
        ("Automatic Robotic Vacuum Cleaner", 299.99, 175.00),
        ("Aroma Oil Diffuser & Humidifier", 29.99, 12.00),
    ],
    "Beauty & Personal Care": [
        ("Vitamin C Brightening Face Serum", 28.99, 8.50),
        ("Hydrating Hyaluronic Acid Cream", 22.99, 6.80),
        ("Sonic Electric Toothbrush", 64.99, 29.00),
        ("Argan Oil Hair Repair Mask", 19.99, 5.50),
        ("Rosewater Revitalizing Facial Mist", 16.99, 4.20),
    ],
    "Books": [
        ("Data Science & Machine Learning Handbook", 44.99, 18.00),
        ("Atomic Habits for Daily Productivity", 21.99, 7.50),
        ("The Psychology of Money & Wealth", 23.99, 8.00),
        ("Deep Work & Focus in a Distracted World", 22.50, 7.20),
        ("Designing Data-Intensive Applications", 54.99, 22.00),
    ]
}

categories = list(products_catalog.keys())
cat_probs = [0.28, 0.24, 0.20, 0.16, 0.12]

chosen_cats = np.random.choice(categories, size=n_records, p=cat_probs)
product_names = []
unit_prices = []
unit_costs = []

for cat in chosen_cats:
    prods = products_catalog[cat]
    prod = prods[np.random.choice(len(prods))]
    product_names.append(prod[0])
    unit_prices.append(prod[1])
    unit_costs.append(prod[2])

quantities = np.random.choice([1, 2, 3, 4, 5], size=n_records, p=[0.50, 0.25, 0.14, 0.07, 0.04])
discounts = np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35], size=n_records, p=[0.35, 0.20, 0.18, 0.12, 0.08, 0.05, 0.02])

# Calculations
total_revenue = [round(q * p * (1 - d), 2) for q, p, d in zip(quantities, unit_prices, discounts)]
total_cost = [round(q * c, 2) for q, c in zip(quantities, unit_costs)]
profit = [round(r - c, 2) for r, c in zip(total_revenue, total_cost)]

payment_methods = np.random.choice(["Credit Card", "Debit Card", "UPI / Digital Wallet", "Cash on Delivery"], size=n_records, p=[0.42, 0.22, 0.26, 0.10])
regions = np.random.choice(["North", "South", "East", "West"], size=n_records, p=[0.30, 0.28, 0.22, 0.20])
txn_ids = [f"TXN-{100001 + i}" for i in range(n_records)]

df = pd.DataFrame({
    "Transaction_ID": txn_ids,
    "Date": [d.strftime("%Y-%m-%d") for d in dates],
    "Customer_ID": assigned_custs,
    "Gender": genders,
    "Age": ages,
    "Age_Group": age_groups,
    "Region": regions,
    "Product_Category": chosen_cats,
    "Product_Name": product_names,
    "Quantity": quantities,
    "Unit_Price": unit_prices,
    "Discount_Pct": discounts,
    "Total_Revenue": total_revenue,
    "Total_Cost": total_cost,
    "Profit": profit,
    "Payment_Method": payment_methods
})

target_csv = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L1-EDARetailSales\data\retail_sales_dataset.csv"
df.to_csv(target_csv, index=False)
print(f"Generated {len(df)} rows to {target_csv}")
print(df.head(3))
