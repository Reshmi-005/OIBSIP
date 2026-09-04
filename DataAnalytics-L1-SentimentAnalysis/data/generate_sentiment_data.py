import numpy as np
import pandas as pd
import random

np.random.seed(42)
random.seed(42)

positive_templates = [
    "Absolutely love this product! The quality exceeded my expectations and delivery was super fast.",
    "Exceptional build quality and battery life. Best purchase I have made this year!",
    "Outstanding customer support. They resolved my query within ten minutes with great politeness.",
    "High quality materials, elegant design, and totally worth every single penny.",
    "Works flawlessly right out of the box. Highly recommended to anyone looking for reliable gear.",
    "I am extremely satisfied with this purchase. The interface is intuitive and response time is instant.",
    "Fantastic value for money! Superior craftsmanship and sleek modern aesthetics.",
    "Super fast shipping! Arrived two days early in pristine condition. Five stars all the way.",
    "Top notch performance. Handles intensive daily use with zero lag or overheating.",
    "Remarkable experience from ordering to delivery. The product works like a dream.",
    "Brilliant design and rock solid durability. I have already recommended it to all my colleagues.",
    "Incredible sound clarity and deep bass. Far better than competitors in this price bracket.",
    "Very easy to set up and configure. Saved me hours of headache. Wonderful product!",
    "A game changer for my daily routine. Elegant, efficient, and exceptionally well engineered.",
    "Delightful customer experience! The packaging was eco-friendly and the item works beautifully."
]

neutral_templates = [
    "The product is okay. It does the job adequately but nothing extraordinary for the price point.",
    "Standard delivery time. Packaging was adequate. Item matches the dimensions listed.",
    "Average performance overall. Battery lasts about five hours as stated in the manual.",
    "Arrived on time. The color is slightly different from the website photos but acceptable.",
    "Decent quality for the cost. Neither particularly impressed nor disappointed with the outcome.",
    "It works as expected for a budget device. Does not stand out in any specific area.",
    "The unit functions normally. Setup took about twenty minutes following the user guide.",
    "Fair product for routine daily tasks. It gets the job done without any fancy features.",
    "Moderate build quality. Meets the basic specifications mentioned in the catalog description.",
    "It is acceptable. Neither great nor terrible, just an everyday standard utility item.",
    "Delivery took a week as scheduled. The item arrived in plain packaging and operates properly.",
    "Practical design, though the plastic casing feels standard. Acceptable performance.",
    "Basic features work fine. Nothing noteworthy to write home about.",
    "The size is as described. Operates within normal operational parameters.",
    "Standard quality for this price range. Serves its intended purpose."
]

negative_templates = [
    "Terrible experience. The item stopped working after two days and customer support was useless.",
    "Very poor build quality. Feels cheap, flimsy, and completely fragile. Do not waste your money.",
    "Extremely disappointed. The package arrived crushed, damaged, and parts were missing.",
    "Dreadful battery life and constant connection drops. Total waste of money and time.",
    "Worst customer service ever encountered. Waited three weeks for delivery and received the wrong model.",
    "Completely useless item. It overheated within fifteen minutes of light usage and shut down.",
    "Misleading product description. The actual item does not support the features advertised.",
    "Horrible quality control. The buttons are stiff, unresponsive, and the screen flickered immediately.",
    "Fell apart after one week of gentle use. Cheap materials and non-existent warranty support.",
    "Unacceptable delays in shipping and zero communication from the vendor. Requesting immediate refund.",
    "Defective on arrival. Power switch did not turn on at all. Total garbage.",
    "Do not buy this product! Extremely frustrated with the buggy software and constant crashes.",
    "Cheap plastic that cracked on first contact. The worst purchase I have made in years.",
    "Terrible customer care. Sent three emails regarding the defect and received zero response.",
    "Absolute nightmare. Flawed design, noisy motor, and terrible instruction manual."
]

# Tricky / nuanced reviews for error analysis
tricky_samples = [
    ("Not bad at all, actually quite impressive compared to cheaper alternatives.", "Positive"),
    ("I really wanted to love this, but unfortunately it broke on day one.", "Negative"),
    ("Great, another delay in shipping just when I needed it urgently.", "Negative"),
    ("The design is gorgeous, but the software is completely unusable and buggy.", "Negative"),
    ("Nothing special, though it might work fine for beginners.", "Neutral"),
    ("Not the worst item I have owned, but certainly far from the best.", "Neutral"),
    ("Surprisingly good for such an inexpensive item, exceeded my doubts.", "Positive"),
    ("Works fine I guess, but I would not buy it a second time.", "Neutral"),
    ("Amazing packaging, but the product inside was totally defective.", "Negative"),
    ("Could be better, could be worse, just an average purchase.", "Neutral")
]

# Generate synthetic variations
n_per_class = 600
reviews = []
sentiments = []

def generate_variation(base_text):
    prefixes = ["", "Honestly, ", "In my opinion, ", "After using it for two weeks: ", "Update: ", "Short review: "]
    suffixes = ["", " Would recommend.", " Glad I bought it.", " Will look for alternatives.", " Keep this in mind.", " Just my two cents."]
    prefix = random.choice(prefixes)
    suffix = random.choice(suffixes)
    return f"{prefix}{base_text}{suffix}".strip()

# Positive
for _ in range(n_per_class):
    base = random.choice(positive_templates)
    reviews.append(generate_variation(base))
    sentiments.append("Positive")

# Neutral
for _ in range(n_per_class):
    base = random.choice(neutral_templates)
    reviews.append(generate_variation(base))
    sentiments.append("Neutral")

# Negative
for _ in range(n_per_class):
    base = random.choice(negative_templates)
    reviews.append(generate_variation(base))
    sentiments.append("Negative")

# Add tricky samples
for text, label in tricky_samples:
    reviews.append(text)
    sentiments.append(label)

df = pd.DataFrame({
    "Review_ID": [f"REV-{10001 + i}" for i in range(len(reviews))],
    "Review_Text": reviews,
    "Sentiment": sentiments
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

target_csv = r"C:\Users\reshm\.gemini\antigravity\scratch\OIBSIP\DataAnalytics-L1-SentimentAnalysis\data\customer_feedback_sentiment.csv"
df.to_csv(target_csv, index=False)
print(f"Generated {len(df)} sentiment records to {target_csv}")
print(df['Sentiment'].value_counts())
print(df.head(3))
