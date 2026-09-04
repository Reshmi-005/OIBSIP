"""
Task 3: Credit Card Fraud Detection (Imbalanced Machine Learning Pipeline)
Track: Data Analytics (Level 2) - OIBSIP
Author: Data Analytics Intern
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "fraud_detection_dataset.csv")
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 70)
    print("TASK 3 - CREDIT CARD FRAUD DETECTION PIPELINE")
    print("=" * 70)

    # 1. Ingestion & Class Imbalance Audit
    print("\n[1] DATA INGESTION & CLASS IMBALANCE AUDIT")
    df = pd.read_csv(data_path)
    total_txns = len(df)
    n_fraud = df['Class'].sum()
    n_legit = total_txns - n_fraud
    fraud_pct = (n_fraud / total_txns) * 100

    print(f"- Total Transactions Analyzed : {total_txns:,}")
    print(f"- Legitimate Transactions (0) : {n_legit:,} ({100 - fraud_pct:.2f}%)")
    print(f"- Fraudulent Transactions (1) : {n_fraud:,} ({fraud_pct:.2f}%)")

    # 2. EDA: Amount Distributions & Hour of Day
    print("\n[2] GENERATING EDA VISUALIZATIONS (AMOUNTS & TIME-OF-DAY)...")
    df['Hour'] = ((df['Time'] / 3600) % 24).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # Class balance bar & amount distribution
    sns.boxplot(x='Class', y='Amount', data=df, ax=axes[0], palette=['#27ae60', '#e74c3c'], hue='Class', legend=False)
    axes[0].set_yscale('log')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Legitimate (0)', 'Fraudulent (1)'], fontweight='bold')
    axes[0].set_title("Transaction Amount Distribution (Log Scale)", fontweight='bold')
    axes[0].set_xlabel("Transaction Class", fontweight='bold')
    axes[0].set_ylabel("Amount ($ - Log Scale)", fontweight='bold')

    # Mean amounts annotation
    legit_mean = df[df['Class'] == 0]['Amount'].mean()
    fraud_mean = df[df['Class'] == 1]['Amount'].mean()
    axes[0].text(0, legit_mean + 50, f"Mean: ${legit_mean:.2f}", ha='center', fontweight='bold', color='#1e8449')
    axes[0].text(1, fraud_mean + 150, f"Mean: ${fraud_mean:.2f}", ha='center', fontweight='bold', color='#922b21')

    # Class counts donut
    axes[1].pie([n_legit, n_fraud], labels=['Legitimate (99.36%)', 'Fraud (0.64%)'],
                colors=['#27ae60', '#e74c3c'], autopct='%1.2f%%', startangle=140,
                wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2), textprops=dict(fontweight='bold'))
    axes[1].set_title("Severe Class Imbalance Breakdown", fontweight='bold')

    plt.tight_layout()
    chart1_path = os.path.join(images_dir, "01_class_imbalance_and_amount_distribution.png")
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"Saved class imbalance and amount chart to: {chart1_path}")

    # Time of Day Analysis
    plt.figure(figsize=(11, 4.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    
    hourly_legit = df[df['Class'] == 0]['Hour'].value_counts(normalize=True).sort_index() * 100
    hourly_fraud = df[df['Class'] == 1]['Hour'].value_counts(normalize=True).sort_index() * 100

    plt.plot(hourly_legit.index, hourly_legit.values, marker='o', color='#27ae60', linewidth=2.2, label='Legitimate Velocity (% of Legit Txns)')
    plt.plot(hourly_fraud.index, hourly_fraud.values, marker='s', color='#e74c3c', linewidth=2.5, linestyle='--', label='Fraud Velocity (% of Fraud Txns)')
    plt.title("Transaction Activity Velocity by Hour of Day (Fraud vs. Legitimate)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Hour of Day (0 - 23 UTC)", fontweight='bold')
    plt.ylabel("Transaction Volume Share (%)", fontweight='bold')
    plt.xticks(range(0, 24))
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart2_path = os.path.join(images_dir, "02_time_of_day_analysis.png")
    plt.savefig(chart2_path, dpi=150)
    plt.close()
    print(f"Saved time-of-day chart to: {chart2_path}")

    # 3. Stratified Train / Test Partitioning
    print("\n[3] STRATIFIED TRAIN / TEST SPLIT (80% TRAIN, 20% TEST)")
    feature_cols = [f"V{i}" for i in range(1, 9)] + ['Amount', 'Hour']
    X = df[feature_cols]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"- Training set: {len(X_train):,} samples (Legit: {(y_train==0).sum():,}, Fraud: {(y_train==1).sum():,})")
    print(f"- Testing set : {len(X_test):,} samples (Legit: {(y_test==0).sum():,}, Fraud: {(y_test==1).sum():,})")

    # 4. Class Imbalance Remediation via SMOTE
    print("\n[4] SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE (SMOTE)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42, sampling_strategy=0.5) # Resample minority to 50% of majority
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

    print(f"- Pre-SMOTE Training Class Counts : Legit = {(y_train==0).sum():,}, Fraud = {(y_train==1).sum():,}")
    print(f"- Post-SMOTE Training Class Counts: Legit = {(y_train_smote==0).sum():,}, Fraud = {(y_train_smote==1).sum():,}")

    # Visual SMOTE comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#f8f9fa')
    
    # Pre-SMOTE scatter (V1 vs V4)
    sns.scatterplot(x=X_train['V1'], y=X_train['V4'], hue=y_train, palette=['#27ae60', '#e74c3c'], ax=axes[0], s=35, alpha=0.7)
    axes[0].set_title("Pre-SMOTE Training Space (Severe Imbalance)", fontweight='bold')
    
    # Post-SMOTE scatter
    sns.scatterplot(x=X_train_smote[:, 0], y=X_train_smote[:, 3], hue=y_train_smote, palette=['#27ae60', '#e74c3c'], ax=axes[1], s=35, alpha=0.5)
    axes[1].set_title("Post-SMOTE Balanced Decision Space", fontweight='bold')
    
    plt.tight_layout()
    chart3_path = os.path.join(images_dir, "03_smote_resampling_comparison.png")
    plt.savefig(chart3_path, dpi=150)
    plt.close()
    print(f"Saved SMOTE comparison chart to: {chart3_path}")

    # 5. Model Training: Logistic Regression & Random Forest
    print("\n[5] TRAINING BALANCED CLASSIFIERS")
    models = {
        "Logistic Regression (SMOTE)": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest (Balanced Weights)": RandomForestClassifier(n_estimators=150, max_depth=8, class_weight='balanced', random_state=42)
    }

    results = []
    y_preds = {}
    y_probs = {}

    # Train LR on SMOTE data
    lr = models["Logistic Regression (SMOTE)"]
    lr.fit(X_train_smote, y_train_smote)
    lr_pred = lr.predict(X_test_scaled)
    lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
    y_preds["Logistic Regression (SMOTE)"] = lr_pred
    y_probs["Logistic Regression (SMOTE)"] = lr_prob

    # Train RF on cost-sensitive balanced weights
    rf = models["Random Forest (Balanced Weights)"]
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    y_preds["Random Forest (Balanced Weights)"] = rf_pred
    y_probs["Random Forest (Balanced Weights)"] = rf_prob

    # 6. Evaluation: Precision, Recall, F1, AUC-ROC
    for name in models.keys():
        pred = y_preds[name]
        prob = y_probs[name]
        acc = accuracy_score(y_test, pred)
        p = precision_score(y_test, pred)
        r = recall_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        auc = roc_auc_score(y_test, prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": p,
            "Recall (Sensitivity)": r,
            "F1-Score": f1,
            "AUC-ROC": auc
        })
        print(f"\n--- {name.upper()} ---")
        print(classification_report(y_test, pred, target_names=['Legitimate', 'Fraud'], digits=4))

    results_df = pd.DataFrame(results)
    print("\nFRAUD DETECTION MODEL BENCHMARK TABLE:")
    print(results_df.round(4).to_string(index=False))

    # 7. Confusion Matrices Heatmap
    print("\n[7] GENERATING CONFUSION MATRICES...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#f8f9fa')
    cm_labels = ['Legitimate (0)', 'Fraud (1)']

    for i, (name, pred) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[i],
                    xticklabels=cm_labels, yticklabels=cm_labels, cbar=False)
        r_val = recall_score(y_test, pred)
        p_val = precision_score(y_test, pred)
        axes[i].set_title(f"{name}\n(Recall: {r_val*100:.1f}% | Precision: {p_val*100:.1f}%)", fontweight='bold', pad=10)
        axes[i].set_xlabel("Predicted Label", fontweight='bold')
        axes[i].set_ylabel("True Label", fontweight='bold')

    plt.tight_layout()
    chart4_path = os.path.join(images_dir, "04_confusion_matrices.png")
    plt.savefig(chart4_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrices to: {chart4_path}")

    # 8. ROC Curves
    plt.figure(figsize=(8, 5.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    
    for name, prob in y_probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        plt.plot(fpr, tpr, linewidth=2.2, label=f"{name} (AUC = {auc:.4f})")
    
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random Guess Baseline')
    plt.title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (Fall-out)", fontweight='bold')
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    chart5_path = os.path.join(images_dir, "05_roc_curves.png")
    plt.savefig(chart5_path, dpi=150)
    plt.close()
    print(f"Saved ROC curves to: {chart5_path}")

    # 9. Feature Importance Ranking
    print("\n[9] GENERATING FEATURE IMPORTANCE RANKING...")
    fi_rf = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(9, 4.8))
    plt.gcf().patch.set_facecolor('#f8f9fa')
    plt.barh(fi_rf['Feature'], fi_rf['Importance'], color='#c0392b', edgecolor='black', linewidth=0.5)
    plt.title("Random Forest Gini Feature Importance in Fraud Detection", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Gini Importance Score", fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    chart6_path = os.path.join(images_dir, "06_feature_importance_ranking.png")
    plt.savefig(chart6_path, dpi=150)
    plt.close()
    print(f"Saved feature importance ranking to: {chart6_path}")

    # 10. Discussion on Scalability
    print("\n" + "=" * 70)
    print("PRODUCTION SCALABILITY: SERVING 1 MILLION TRANSACTIONS / HOUR")
    print("=" * 70)
    print("""
Architectural Blueprint for 1M Txns / Hour (~280 TPS steady, ~1,200 TPS peak):
1. Low-Latency Inference:
   - Export Random Forest / Logistic Regression models to ONNX (Open Neural Network Exchange) or Treelite C-arrays, yielding sub-2 millisecond inference times.
2. Real-Time Streaming Pipeline:
   - Kafka / Apache Pulsar for transaction event streaming.
   - Apache Flink for real-time stateful feature aggregation (e.g. cardholder spend velocity in last 10 mins).
3. In-Memory Low-Latency Feature Store:
   - Redis Cluster / Feast to serve cardholder historical velocity profiles with sub-5ms lookup latency.
    """)

if __name__ == "__main__":
    main()
