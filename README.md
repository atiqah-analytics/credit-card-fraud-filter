# Dual-Layer Credit Card Fraud Detection Pipelin
## 📌 Project Overview
This project implements a production-ready, hybrid fraud detection system designed to handle real-world credit card transactions. Rather than relying solely on machine learning, the architecture employs a two-tier screening filter to effectively balance financial protection against user friction.



## 🛠️ Architecture & Filter Design
* **Layer 1: Deterministic Rule Engine:** Catches macro-anomalies instantly based on preset business thresholds (e.g., transaction amounts > $5,000) before triggering complex models, saving computational overhead.
* **Layer 2: Probabilistic ML Classifier:** Uses a fast, tree-based **LightGBM** model to extract subtle patterns from anonymized feature sets ($V1$ through $V28$) to spot highly-disguised fraud.

## 📊 Overcoming Extreme Data Imbalance
Fraud represents less than **0.17%** of this dataset (492 fraud cases out of 284,807 transactions). Standard accuracy is a deceptive metric here, as a broken model guessing "Not Fraud" 100% of the time would achieve 99.83% accuracy while catching zero fraud.

To solve this class imbalance, this pipeline utilizes **cost-sensitive learning (`class_weight='balanced'`)** to penalize false negatives heavily, optimizing the system for high **Recall**.

## 📈 Performance & Business Metrics
The complete hybrid filter achieved the following results on the unseen test set:

* **Fraud Caught (True Positives):** 87 cases
* **Fraud Missed (False Negatives):** 11 cases (~89% Recall)
* **Innocent Users Flagged (False Positives):** Moderate user friction, easily handled by standard SMS automated verification loops.

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR-USERNAME/credit-card-fraud-filter.git](https://github.com/YOUR-USERNAME/credit-card-fraud-filter.git)
