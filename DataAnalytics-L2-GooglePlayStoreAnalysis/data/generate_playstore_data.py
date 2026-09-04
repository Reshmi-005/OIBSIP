import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

base_dir = os.path.dirname(os.path.abspath(__file__))

categories = [
    'FAMILY', 'GAME', 'TOOLS', 'PRODUCTIVITY', 'MEDICAL', 
    'FINANCE', 'COMMUNICATION', 'PHOTOGRAPHY', 'LIFESTYLE', 
    'BUSINESS', 'HEALTH_AND_FITNESS', 'PERSONALIZATION'
]

genres_map = {
    'FAMILY': ['Casual', 'Education', 'Entertainment', 'Puzzle'],
    'GAME': ['Action', 'Arcade', 'Strategy', 'Role Playing', 'Racing', 'Card'],
    'TOOLS': ['Tools'],
    'PRODUCTIVITY': ['Productivity'],
    'MEDICAL': ['Medical'],
    'FINANCE': ['Finance'],
    'COMMUNICATION': ['Communication'],
    'PHOTOGRAPHY': ['Photography'],
    'LIFESTYLE': ['Lifestyle'],
    'BUSINESS': ['Business'],
    'HEALTH_AND_FITNESS': ['Health & Fitness'],
    'PERSONALIZATION': ['Personalization']
}

install_tiers = [
    (10, "10+"), (50, "50+"), (100, "100+"), (500, "500+"),
    (1000, "1,000+"), (5000, "5,000+"), (10000, "10,000+"),
    (50000, "50,000+"), (100000, "100,000+"), (500000, "500,000+"),
    (1000000, "1,000,000+"), (5000000, "5,000,000+"),
    (10000000, "10,000,000+"), (50000000, "50,000,000+"),
    (100000000, "100,000,000+"), (1000000000, "1,000,000,000+")
]

prefixes = ['Pro', 'Super', 'Smart', 'Quick', 'Easy', 'Pocket', 'Hyper', 'Master', 'Daily', 'Global', 'NextGen', 'Fast']
suffixes = ['Tracker', 'Hub', 'Manager', 'Plus', 'Hero', 'Lite', 'Assistant', 'Studio', 'Portal', 'Vault', 'Flow', '3D']

apps = []
n_apps = 2500

for i in range(n_apps):
    cat = random.choice(categories)
    genre = random.choice(genres_map[cat])
    name = f"{random.choice(prefixes)} {random.choice(suffixes)} {cat.title()[:4]} {i+1}"
    
    # Category install probabilities
    if cat in ['GAME', 'COMMUNICATION']:
        weights = [0.01]*4 + [0.03]*4 + [0.08]*4 + [0.15, 0.20, 0.15, 0.08]
    elif cat in ['PRODUCTIVITY', 'TOOLS', 'PHOTOGRAPHY']:
        weights = [0.02]*4 + [0.06]*4 + [0.12]*4 + [0.08, 0.08, 0.05, 0.01]
    else:
        weights = [0.05]*4 + [0.10]*4 + [0.08]*4 + [0.04, 0.03, 0.01, 0.00]
    
    # normalize
    weights = [w / sum(weights) for w in weights]
    tier_idx = np.random.choice(len(install_tiers), p=weights)
    numeric_installs, install_str = install_tiers[tier_idx]
    
    # Reviews
    rev_ratio = random.uniform(0.01, 0.08)
    numeric_reviews = max(1, int(numeric_installs * rev_ratio))
    reviews_str = str(numeric_reviews)
    
    # Rating (realistic skewed around 4.1 to 4.4, with ~10% NaNs)
    if random.random() < 0.10:
        rating = np.nan
    else:
        rating = round(min(5.0, max(1.0, np.random.normal(4.18, 0.52))), 1)
    
    # Size
    size_rand = random.random()
    if size_rand < 0.12:
        size_str = "Varies with device"
    elif size_rand < 0.18:
        size_kb = random.randint(150, 950)
        size_str = f"{size_kb}k"
    else:
        if cat == 'GAME':
            size_mb = round(random.uniform(15, 95), 1)
        else:
            size_mb = round(random.uniform(2, 45), 1)
        size_str = f"{size_mb}M"
        
    # Free vs Paid (~92% Free, ~8% Paid)
    is_paid = random.random() < 0.08
    if is_paid:
        app_type = 'Paid'
        # Check for outlier 'I am Rich' type
        if random.random() < 0.03:
            price_val = random.choice([299.99, 379.99, 399.99])
        else:
            price_val = random.choice([0.99, 1.49, 1.99, 2.99, 3.99, 4.99, 7.99, 9.99, 14.99])
        price_str = f"${price_val:.2f}"
    else:
        app_type = 'Free'
        price_str = "$0"
        
    content_rating = random.choice(['Everyone', 'Everyone', 'Everyone', 'Teen', 'Mature 17+', 'Everyone 10+'])
    last_updated = f"{random.choice(['January', 'March', 'May', 'July', 'August', 'October'])} {random.randint(1, 28)}, 2024"
    cur_ver = f"{random.randint(1, 5)}.{random.randint(0, 9)}.{random.randint(0, 9)}"
    android_ver = random.choice(['4.1 and up', '4.4 and up', '5.0 and up', '6.0 and up', '7.0 and up', '8.0 and up'])
    
    apps.append({
        'App': name,
        'Category': cat,
        'Rating': rating,
        'Reviews': reviews_str,
        'Size': size_str,
        'Installs': install_str,
        'Type': app_type,
        'Price': price_str,
        'Content Rating': content_rating,
        'Genres': genre,
        'Last Updated': last_updated,
        'Current Ver': cur_ver,
        'Android Ver': android_ver
    })

# Add the iconic luxury outlier
apps.append({
    'App': 'I Am Rich Luxury Edition',
    'Category': 'LIFESTYLE',
    'Rating': 3.8,
    'Reviews': '54',
    'Size': '12M',
    'Installs': '100+',
    'Type': 'Paid',
    'Price': '$399.99',
    'Content Rating': 'Everyone',
    'Genres': 'Lifestyle',
    'Last Updated': 'June 15, 2024',
    'Current Ver': '1.0.0',
    'Android Ver': '5.0 and up'
})

df_apps = pd.DataFrame(apps)
apps_csv_path = os.path.join(base_dir, "googleplaystore.csv")
df_apps.to_csv(apps_csv_path, index=False)
print(f"Generated {len(df_apps)} apps in: {apps_csv_path}")

# Generate Reviews Dataset
reviews = []
sample_apps = df_apps.sample(n=350, random_state=42)['App'].tolist()

positive_phrases = [
    "Absolutely love this app! Very smooth UI and responsive.",
    "Best in its class, saves me so much time every day.",
    "Incredible features and clean layout. Worth every penny.",
    "Game is highly addictive with stellar graphics!",
    "Great customer support and seamless synchronization.",
    "Essential utility for daily workflow. Fast loading."
]

negative_phrases = [
    "Constant crashes after the latest update. Terrible experience.",
    "Too many intrusive video ads every 30 seconds. Uninstalled.",
    "Battery drain is immense and phone overheats quickly.",
    "Subscription fee is ridiculous for basic features that used to be free.",
    "Full of bugs, login screen freezes constantly.",
    "Customer service never responded to my refund ticket."
]

neutral_phrases = [
    "Decent app overall, does the job but lacks advanced customization.",
    "Average performance. UI could use some polishing.",
    "Good concept, waiting for future feature updates.",
    "Works okay on tablet, bit clunky on small screens."
]

for app in sample_apps:
    n_revs = random.randint(5, 12)
    for _ in range(n_revs):
        p_type = random.random()
        if p_type < 0.60:
            text = random.choice(positive_phrases)
            sentiment = 'Positive'
            polarity = round(random.uniform(0.25, 0.95), 3)
            subjectivity = round(random.uniform(0.40, 0.85), 3)
        elif p_type < 0.82:
            text = random.choice(negative_phrases)
            sentiment = 'Negative'
            polarity = round(random.uniform(-0.85, -0.15), 3)
            subjectivity = round(random.uniform(0.45, 0.90), 3)
        else:
            text = random.choice(neutral_phrases)
            sentiment = 'Neutral'
            polarity = round(random.uniform(-0.10, 0.15), 3)
            subjectivity = round(random.uniform(0.15, 0.40), 3)
            
        reviews.append({
            'App': app,
            'Translated_Review': text,
            'Sentiment': sentiment,
            'Sentiment_Polarity': polarity,
            'Sentiment_Subjectivity': subjectivity
        })

df_reviews = pd.DataFrame(reviews)
reviews_csv_path = os.path.join(base_dir, "googleplaystore_user_reviews.csv")
df_reviews.to_csv(reviews_csv_path, index=False)
print(f"Generated {len(df_reviews)} user reviews in: {reviews_csv_path}")
