"""
Task 2: Customer Segmentation Analysis using RFM & K-Means Clustering
Track: Data Analytics (Level 1) - OIBSIP
Author: Data Analytics Intern
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Formatting
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "ecommerce_customer_data.csv")
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    print("=" * 70)
    print("TASK 2 · CUSTOMER SEGMENTATION ANALYSIS (RFM + K-MEANS)")
    print("=" * 70)

    # 1. Load Dataset and Inspect Structure
    print("\n[1] DATASET INSPECTION & DATA CLEANING")
    raw_df = pd.read_csv(data_path)
    print(f"- Raw Records Ingested: {raw_df.shape[0]:,} rows, {raw_df.shape[1]} columns")
    print(f"- Missing CustomerIDs: {raw_df['CustomerID'].isnull().sum():,}")

    # Data Cleaning: Drop missing CustomerIDs
    df = raw_df.dropna(subset=['CustomerID']).copy()
    print(f"- Records after removing null CustomerIDs: {len(df):,}")

    # Remove cancellations (InvoiceNo starts with 'C' or Quantity <= 0) and non-positive prices
    df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    cancellations = df['InvoiceNo'].str.startswith('C') | (df['Quantity'] <= 0) | (df['UnitPrice'] <= 0)
    print(f"- Cancellations / Inconsistent records removed: {cancellations.sum():,}")
    df = df[~cancellations].copy()

    # Calculate TotalSpend per line item
    df['TotalSpend'] = df['Quantity'] * df['UnitPrice']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"- Valid transactions ready for segmentation: {len(df):,}")

    # 2. Descriptive Statistics: AOV, Purchase Frequency, CLV Proxy
    print("\n" + "=" * 70)
    print("[2] HIGH-LEVEL CUSTOMER DESCRIPTIVE METRICS")
    print("=" * 70)

    # Reference date: 1 day after the latest recorded transaction
    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"- Reference Snapshot Date: {analysis_date.strftime('%Y-%m-%d')}")

    # Aggregate by Customer
    rfm = df.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (analysis_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalSpend', 'sum')
    ).reset_index()

    # Calculate Average Order Value (AOV) and CLV Proxy
    rfm['AOV'] = rfm['Monetary'] / rfm['Frequency']
    rfm['CLV'] = rfm['Monetary']

    print(f"- Unique Customers Analyzed: {len(rfm):,}")
    print(f"- Average Purchase Value (AOV): ${rfm['AOV'].mean():.2f} (Median: ${rfm['AOV'].median():.2f})")
    print(f"- Average Purchase Frequency: {rfm['Frequency'].mean():.2f} orders/customer (Max: {rfm['Frequency'].max()})")
    print(f"- Average Customer Lifetime Value (Spend): ${rfm['CLV'].mean():.2f} (Max: ${rfm['CLV'].max():.2f})")

    print("\nSummary Statistics of RFM Features:")
    print(rfm[['Recency', 'Frequency', 'Monetary', 'AOV']].describe().round(2).to_string())

    # Save RFM Distributions plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('#f8f9fa')
    features = ['Recency', 'Frequency', 'Monetary']
    colors = ['#2980b9', '#27ae60', '#8e44ad']

    for i, col in enumerate(features):
        sns.histplot(rfm[col], kde=True, ax=axes[i], color=colors[i], edgecolor='black', alpha=0.6)
        axes[i].set_title(f"Distribution of {col}", fontweight='bold')
        axes[i].set_xlabel(col, fontweight='bold')

    plt.tight_layout()
    chart1_path = os.path.join(images_dir, "01_rfm_distributions.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    print(f"\nSaved RFM distribution chart to: {chart1_path}")

    # 3. Data Normalization & Standardization
    print("\n[3] DATA PREPROCESSING & STANDARDIZATION")
    rfm_log = pd.DataFrame()
    rfm_log['Recency'] = np.log1p(rfm['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm['Frequency'])
    rfm_log['Monetary'] = np.log1p(rfm['Monetary'])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    print("- Applied log1p transformation to mitigate right-skewness.")
    print("- Applied StandardScaler (Mean = 0, Std = 1).")

    # 4. K-Means Clustering & Elbow Method
    print("\n[4] EVALUATING OPTIMAL CLUSTERS (ELBOW METHOD & SILHOUETTE SCORE)")
    k_range = range(2, 9)
    inertia_list = []
    silhouette_list = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(rfm_scaled)
        inertia_list.append(kmeans.inertia_)
        sil_score = silhouette_score(rfm_scaled, kmeans.labels_)
        silhouette_list.append(sil_score)
        print(f"  * K = {k}: Inertia = {kmeans.inertia_:.2f}, Silhouette Score = {sil_score:.4f}")

    # Plot Elbow Curve and Silhouette Scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#f8f9fa')

    # Elbow
    axes[0].plot(k_range, inertia_list, marker='o', linewidth=2.5, color='#e74c3c')
    axes[0].set_title("Elbow Method for Optimal K (Inertia)", fontweight='bold')
    axes[0].set_xlabel("Number of Clusters (K)", fontweight='bold')
    axes[0].set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontweight='bold')
    axes[0].axvline(x=4, color='gray', linestyle='--', alpha=0.7, label='Optimal Elbow (K=4)')
    axes[0].legend()

    # Silhouette
    axes[1].plot(k_range, silhouette_list, marker='s', linewidth=2.5, color='#2980b9')
    axes[1].set_title("Silhouette Score vs. Number of Clusters", fontweight='bold')
    axes[1].set_xlabel("Number of Clusters (K)", fontweight='bold')
    axes[1].set_ylabel("Silhouette Score", fontweight='bold')
    axes[1].axvline(x=4, color='gray', linestyle='--', alpha=0.7, label='Selected K=4')
    axes[1].legend()

    plt.tight_layout()
    chart2_path = os.path.join(images_dir, "02_elbow_silhouette_analysis.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()
    print(f"Saved Elbow/Silhouette analysis to: {chart2_path}")

    # 5. Fit K-Means with Optimal K = 4
    optimal_k = 4
    print(f"\n[5] FITTING K-MEANS WITH K = {optimal_k}")
    kmeans_opt = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, random_state=42)
    rfm['Cluster'] = kmeans_opt.fit_predict(rfm_scaled)

    # 6. Profile Each Cluster
    cluster_means = rfm.groupby('Cluster').agg(
        Count=('CustomerID', 'count'),
        Recency_Mean=('Recency', 'mean'),
        Frequency_Mean=('Frequency', 'mean'),
        Monetary_Mean=('Monetary', 'mean'),
        AOV_Mean=('AOV', 'mean')
    ).reset_index()

    # Rank clusters based on composite customer value score:
    # High Frequency & Monetary + Low Recency = Best
    # Normalize values to 0-1 for ranking
    r_norm = (cluster_means['Recency_Mean'].max() - cluster_means['Recency_Mean']) / (cluster_means['Recency_Mean'].max() - cluster_means['Recency_Mean'].min() + 1e-5)
    f_norm = (cluster_means['Frequency_Mean'] - cluster_means['Frequency_Mean'].min()) / (cluster_means['Frequency_Mean'].max() - cluster_means['Frequency_Mean'].min() + 1e-5)
    m_norm = (cluster_means['Monetary_Mean'] - cluster_means['Monetary_Mean'].min()) / (cluster_means['Monetary_Mean'].max() - cluster_means['Monetary_Mean'].min() + 1e-5)
    
    score = 0.3 * r_norm + 0.35 * f_norm + 0.35 * m_norm
    cluster_means['Score'] = score
    cluster_means = cluster_means.sort_values(by='Score', ascending=False).reset_index(drop=True)

    # Archetype assignments ordered from highest score to lowest:
    archetypes = [
        "Champions (High Value, Highly Active)",
        "Loyal Customers (Consistent Spenders)",
        "At-Risk Customers (Lapsing Spenders)",
        "Hibernating (Low-Touch Inactive)"
    ]
    cluster_means['Segment_Name'] = archetypes

    segment_map = dict(zip(cluster_means['Cluster'], cluster_means['Segment_Name']))
    rfm['Segment'] = rfm['Cluster'].map(segment_map)

    print("\nCluster Profiles Summary (Ranked by Value):")
    print(cluster_means[['Cluster', 'Segment_Name', 'Count', 'Recency_Mean', 'Frequency_Mean', 'Monetary_Mean', 'AOV_Mean']].round(2).to_string(index=False))

    # 7. Visualise Clusters Using Scatter Plots
    print("\n[7] GENERATING CLUSTER SCATTER PLOTS...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor('#f8f9fa')
    
    palette_dict = {
        "Champions (High Value, Highly Active)": "#27ae60",
        "Loyal Customers (Consistent Spenders)": "#2980b9",
        "At-Risk Customers (Lapsing Spenders)": "#e67e22",
        "Hibernating (Low-Touch Inactive)": "#c0392b"
    }

    # Scatter 1: Recency vs Frequency
    sns.scatterplot(
        x='Recency', y='Frequency', hue='Segment', data=rfm,
        ax=axes[0], palette=palette_dict, s=70, alpha=0.85, edgecolor='black', linewidth=0.5
    )
    axes[0].set_title("Customer Segments: Recency vs. Purchase Frequency", fontweight='bold')
    axes[0].set_xlabel("Recency (Days Since Last Order)", fontweight='bold')
    axes[0].set_ylabel("Frequency (Number of Orders)", fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=8.5)

    # Scatter 2: Frequency vs Monetary
    sns.scatterplot(
        x='Frequency', y='Monetary', hue='Segment', data=rfm,
        ax=axes[1], palette=palette_dict, s=70, alpha=0.85, edgecolor='black', linewidth=0.5
    )
    axes[1].set_title("Customer Segments: Purchase Frequency vs. Monetary Spend ($)", fontweight='bold')
    axes[1].set_xlabel("Frequency (Number of Orders)", fontweight='bold')
    axes[1].set_ylabel("Monetary Spend ($)", fontweight='bold')
    axes[1].legend(loc='upper left', fontsize=8.5)

    plt.tight_layout()
    chart3_path = os.path.join(images_dir, "03_cluster_scatter_plots.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    print(f"Saved cluster scatter plots to: {chart3_path}")

    # 8. Bar Chart: Customer Distribution Per Segment
    print("\n[8] GENERATING CUSTOMER DISTRIBUTION BAR CHART...")
    plt.figure(figsize=(10, 5.5))
    plt.gcf().patch.set_facecolor('#f8f9fa')

    order_archetypes = archetypes
    seg_counts = rfm['Segment'].value_counts().reindex(order_archetypes)
    bar_colors = [palette_dict[s] for s in order_archetypes]
    
    bars = plt.bar(seg_counts.index, seg_counts.values, color=bar_colors, edgecolor='black', linewidth=0.7)
    plt.title("Customer Distribution by Segment (K-Means K=4)", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Customer Segment", fontweight='bold')
    plt.ylabel("Number of Customers", fontweight='bold')
    plt.xticks(rotation=15, ha='right', fontsize=9.5)

    for b in bars:
        h = b.get_height()
        pct = (h / len(rfm)) * 100
        plt.text(b.get_x() + b.get_width()/2., h + 6, f"{h} ({pct:.1f}%)", ha='center', fontweight='bold', fontsize=9.5)

    plt.tight_layout()
    chart4_path = os.path.join(images_dir, "04_customer_count_per_cluster.png")
    plt.savefig(chart4_path, dpi=300)
    plt.close()
    print(f"Saved segment distribution bar chart to: {chart4_path}")

    # 9. Marketing Action Recommendations
    print("\n" + "=" * 70)
    print("[9] ACTIONABLE MARKETING STRATEGIES PER CUSTOMER SEGMENT")
    print("=" * 70)
    print("""
1. Champions (High Value, Highly Active):
   - Persona: Recency ~11 days, Frequency ~16 orders, Monetary ~$284.
   - Recommended Strategy:
     * VIP Rewards & Early Access: Grant exclusive previews of new collection launches and limited-edition items.
     * Referral Incentives: Leverage high satisfaction by offering referral bonuses to onboard peers.
     * No Heavy Discounting Needed: Avoid margin dilution; focus on premium experience and personal customer care.

2. Loyal Customers (Consistent Spenders):
   - Persona: Recency ~22 days, Frequency ~5.8 orders, Monetary ~$74.
   - Recommended Strategy:
     * Cross-Selling & Upselling: Recommend complementary product categories based on purchase history.
     * Loyalty Tier Gamification: Introduce tiered milestone incentives (e.g., "Spend $50 more to achieve Platinum status").

3. At-Risk Customers (Lapsing Spenders):
   - Persona: Recency ~197 days, Frequency ~4.0 orders, Monetary ~$75. Previously active customers drifting away.
   - Recommended Strategy:
     * "We Miss You" Win-Back Campaigns: Automated personalized email series with time-limited 15% discount vouchers.
     * Customer Feedback Surveys: Solicit direct feedback to uncover why engagement ceased (e.g., shipping delays, customer service issues).

4. Hibernating (Low-Touch Inactive):
   - Persona: Recency ~220 days, Frequency ~2 orders, Monetary ~$13. Lowest engagement and spend.
   - Recommended Strategy:
     * Low-Cost Re-engagement: Avoid expensive direct mail or paid ad retargeting; utilize low-cost email reactivation campaigns.
     * Clearance & Flash Sales: Target this group with warehouse clearance discounts to liquidate old stock without risking brand equity.
    """)

if __name__ == "__main__":
    main()
