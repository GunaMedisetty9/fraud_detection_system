# model_training.py
import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATA_PATH = "data/creditcard.csv"

# Speed controls
SMOTE_STRATEGY = 0.2          # minority will become 20% of majority (much faster than full balance)
GB_SAMPLE_SIZE = 80000        # train GradientBoosting on only this many rows
RANDOM_STATE = 42


def train_and_save_models():
    """Train multiple models and save them (fast + stable)."""

    print("=" * 60)
    print("FINANCIAL FRAUD DETECTION - MODEL TRAINING (FAST + STABLE)")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    # Load data
    print("\n📂 Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

    # Prepare features/target
    X = df.drop("Class", axis=1)
    y = df["Class"].astype(int)

    # Split
    print("\n✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set:  {X_test.shape[0]} samples")

    # Scale
    print("\n⚖️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, "models/scaler.pkl")
    print("   ✅ Scaler saved: models/scaler.pkl")

    # SMOTE (reduced) to handle imbalance without exploding dataset
    print(f"\n🔄 Applying SMOTE (sampling_strategy={SMOTE_STRATEGY})...")
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=SMOTE_STRATEGY)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    y_train_balanced = np.array(y_train_balanced)  # ensure numpy array for indexing
    print(f"   Balanced training set: {X_train_balanced.shape[0]} samples")
    print(f"   Fraud ratio after SMOTE: {(y_train_balanced.mean()*100):.2f}%")

    # Models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist"
        ),
        # Lighter + trained on subset to avoid long runtimes
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n🤖 Training {name}...")

        if name == "Gradient Boosting":
            # Train GB on a subset to keep it fast
            n = min(GB_SAMPLE_SIZE, X_train_balanced.shape[0])
            rng = np.random.RandomState(RANDOM_STATE)
            idx = rng.choice(X_train_balanced.shape[0], size=n, replace=False)

            model.fit(X_train_balanced[idx], y_train_balanced[idx])
            print(f"   (GB trained on subset: {n} samples)")
        else:
            model.fit(X_train_balanced, y_train_balanced)

        # Evaluate on original test set (not SMOTE)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Save model
        model_filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_filename)
        print(f"   ✅ Model saved: {model_filename}")

        print("\n   Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

        results[name] = {"pred": y_pred, "prob": y_prob}

    # Save test data for dashboard pages
    test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_data["Actual"] = y_test.values
    test_data.to_csv("models/test_data.csv", index=False)

    # Save predictions for Streamlit pages (expected column names)
    predictions_df = pd.DataFrame({
        "Actual": y_test.values,
        "LR_Pred": results["Logistic Regression"]["pred"],
        "LR_Prob": results["Logistic Regression"]["prob"],
        "RF_Pred": results["Random Forest"]["pred"],
        "RF_Prob": results["Random Forest"]["prob"],
        "XGB_Pred": results["XGBoost"]["pred"],
        "XGB_Prob": results["XGBoost"]["prob"],
        "GB_Pred": results["Gradient Boosting"]["pred"],
        "GB_Prob": results["Gradient Boosting"]["prob"],
    })
    predictions_df.to_csv("models/predictions.csv", index=False)

    print("\n" + "=" * 60)
    print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("   Saved: models/*.pkl, models/scaler.pkl, models/test_data.csv, models/predictions.csv")
    print("=" * 60)


if __name__ == "__main__":
    train_and_save_models()