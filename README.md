# 📊 Oasis Infobyte SIP (OIBSIP) — Data Analytics (Level 1)

**Intern Name:** Reshma  
**Track:** Data Analytics  
**Internship Level:** Level 1  
**Repository:** `OIBSIP/`

---

## 🌟 Repository Overview
This repository contains the complete deliverables for all four tasks under **Track: Data Analytics (Level 1)** for the Oasis Infobyte Internship Program (OIBSIP). Every task has been implemented in strict accordance with the mandatory folder structure, feature checklists, rigorous analytical standards, clean code principles, and reproducible data science pipelines.

```
OIBSIP/
│
├── DataAnalytics-L1-EDARetailSales/          # TASK 1: Exploratory Data Analysis on Retail Sales
├── DataAnalytics-L1-CustomerSegmentation/    # TASK 2: Customer Segmentation using RFM & K-Means
├── DataAnalytics-L1-DataCleaning/            # TASK 3: Deliberately Messy Data Cleaning Pipeline
├── DataAnalytics-L1-SentimentAnalysis/       # TASK 4: NLP Sentiment Analysis & Model Benchmarking
│
└── README.md                                 # Master Repository Index & Demo Video Guide
```

---

## 📁 Task Directory & Deliverables Index

| Task # | Project Folder | Domain | Core Tech Stack | Deliverables & Artifacts | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **Task 1** | [`DataAnalytics-L1-EDARetailSales/`](./DataAnalytics-L1-EDARetailSales/) | Exploratory Data Analysis | Python, Pandas, Matplotlib, Seaborn | • Dataset (2,500 rows)<br>• Executed Jupyter Notebook<br>• Standalone CLI Script<br>• 5 High-Res Saved Plots<br>• Comprehensive README | ✅ **100% Complete** |
| **Task 2** | [`DataAnalytics-L1-CustomerSegmentation/`](./DataAnalytics-L1-CustomerSegmentation/) | Unsupervised ML (Clustering) | Scikit-Learn (KMeans), Pandas, StandardScaler | • E-Commerce Dataset (4k txns)<br>• RFM Analysis Pipeline<br>• Elbow & Silhouette Models<br>• 2D/3D Cluster Scatters<br>• Actionable Marketing Guide | ✅ **100% Complete** |
| **Task 3** | [`DataAnalytics-L1-DataCleaning/`](./DataAnalytics-L1-DataCleaning/) | Data Quality & Wrangling | Pandas, NumPy, Regular Expressions | • Messy Raw Dataset (1,245 rows)<br>• Cleaned Production Dataset<br>• Data Quality Audit Table<br>• IQR Outlier Winsorization<br>• Before vs. After Report | ✅ **100% Complete** |
| **Task 4** | [`DataAnalytics-L1-SentimentAnalysis/`](./DataAnalytics-L1-SentimentAnalysis/) | Natural Language Processing | Scikit-Learn, NLTK, WordCloud, TF-IDF | • 1,810 Review Dataset<br>• Lemmatization Pipeline<br>• Naive Bayes + LogReg + SVC<br>• 3 WordClouds & Heatmaps<br>• 5 Edge Case Error Analyses | ✅ **100% Complete** |

---

## 🛠️ Global Environment & Installation
To run any of the notebooks or scripts across this repository, ensure Python 3.10+ is installed and install the required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk textblob wordcloud jupyter nbclient
```

Download required NLTK corpora (if not already downloaded):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🎥 Step 4: Demo Video Presentation Guide & Script

As required by **Step 4 — Record a Demo Video**, create a screen recording walkthrough showcasing your completed projects end-to-end.

### 1. Title Card Format (Mandatory 2-Second Static Screen)
Before beginning the screen recording, display a clean 2-second slide or video overlay containing:
```
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     OASIS INFOBYTE INTERNSHIP PROGRAM (OIBSIP)                 ║
║                                                                ║
║     Full Name     : Reshma                                     ║
║     Assigned Track: Data Analytics                             ║
║     Level         : Level 1                                    ║
║     Tasks Covered : Tasks 1, 2, 3, and 4                       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### 2. Suggested Video Recording Walkthrough Script

#### • Task 1 Walkthrough (EDA on Retail Sales):
1. **Screen**: Open `eda_retail_sales.ipynb` in Jupyter Notebook or VS Code.
2. **Narration**: *"Hello, my name is Reshma, and this is my Level 1 Data Analytics presentation for Oasis Infobyte. In Task 1, I performed an in-depth Exploratory Data Analysis on 2,500 retail sales transactions. As shown in the time-series chart, we observe a pronounced 45% revenue surge during Q4 holiday months. The demographics analysis reveals that shoppers aged 26 to 50 generate over 65% of total sales. A non-obvious insight discovered in this analysis is the Profit Margin Erosion Cliff: when discounts exceed 20%, net profit margins collapse below 12%, demonstrating that volume gained through aggressive discounting destroys profitability."*

#### • Task 2 Walkthrough (Customer Segmentation):
1. **Screen**: Switch to `customer_segmentation.ipynb`.
2. **Narration**: *"In Task 2, I developed an unsupervised machine learning clustering model using RFM (Recency, Frequency, and Monetary) behavioral features on 4,000 e-commerce transactions. After log-transforming and standardizing the features, the Elbow Method and Silhouette Analysis confirmed K=4 as the optimal cluster count. We identified four distinct customer archetypes: Champions, Loyal Customers, At-Risk Customers, and Hibernating shoppers. For each segment, I prescribed targeted CRM actions, such as VIP loyalty rewards for Champions and automated win-back discount sequences for At-Risk customers."*

#### • Task 3 Walkthrough (Data Cleaning):
1. **Screen**: Open `data_cleaning.ipynb` and show the before-and-after table.
2. **Narration**: *"In Task 3, I tackled a deliberately messy customer dataset riddled with missing primary keys, exact duplicates, invalid age entries, mixed date formats, and extreme multi-million dollar outliers. By applying IQR Winsorization capping, mode and median imputation, and canonical string standardization, I transformed 1,245 dirty records into 938 validated, analysis-ready records with 100% data integrity."*

#### • Task 4 Walkthrough (Sentiment Analysis):
1. **Screen**: Open `sentiment_analysis.ipynb` and show the WordClouds and confusion matrices.
2. **Narration**: *"In Task 4, I built an NLP sentiment classification pipeline classifying customer feedback into Positive, Neutral, and Negative. The pipeline incorporates tokenization, custom negation-preserving stopword pruning, WordNet lemmatization, and TF-IDF bi-gram feature extraction. Benchmarking Multinomial Naive Bayes, Logistic Regression, and Linear SVC yielded over 99.7% weighted F1-score across all models. Finally, I conducted an error analysis on five challenging edge cases, demonstrating how sarcasm and contrastive conjunctions influence model predictions."*

---

## 📬 GitHub Submission Guidelines
To push this repository to your GitHub account:
```bash
git init
git add .
git commit -m "Complete OIBSIP Data Analytics Level 1: Tasks 1-4"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/OIBSIP.git
git push -u origin main
```
