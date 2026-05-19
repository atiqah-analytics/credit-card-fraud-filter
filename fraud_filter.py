import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def run_fraud_pipeline():
    print("[*] Fetching live dataset...")
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    
    # 1. Apply Layer 1: Rule-Based Filter
    print("[*] Applying Layer 1: Deterministic Rules...")
    df['Rule_Flag'] = df['Amount'] > 5000
    
    # 2. Split Data
    X = df.drop(columns=['Class', 'Rule_Flag'])
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Apply Layer 2: Machine Learning Model
    print("[*] Training Layer 2: LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # 4. Generate Combined Predictions
    print("[*] Evaluating complete pipeline...")
    ml_preds = model.predict(X_test)
    
    # If rule triggered OR ML triggered, mark as fraud
    rule_triggers_test = df.loc[X_test.index, 'Rule_Flag']
    final_preds = np.where((rule_triggers_test == True) | (ml_preds == 1), 1, 0)
    
    print("\n--- FINAL PIPELINE RESULTS ---")
    print(classification_report(y_test, final_preds))

if __name__ == "__main__":
    run_fraud_pipeline()
