import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import joblib


train_df = pd.read_csv("dataset/master_alphabets_train.csv")
test_df = pd.read_csv("dataset/master_alphabets_test.csv")

TARGET_COLUMN = "label"

X_train = train_df.drop(TARGET_COLUMN, axis=1)
y_train = train_df[TARGET_COLUMN]

X_test = test_df.drop(TARGET_COLUMN, axis=1)
y_test = test_df[TARGET_COLUMN]



le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)



# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=208,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features="log2",
    n_jobs=-1,
    random_state=42
)

# XGBoost
xgb_model = XGBClassifier(
    n_estimators=293,
    max_depth=4,
    learning_rate=0.0698,
    subsample=0.784,
    colsample_bytree=0.988,
    objective="multi:softprob",   # IMPORTANT for soft voting
    eval_metric="mlogloss",
    random_state=42
)

# CatBoost
cat_model = CatBoostClassifier(
    iterations=372,
    depth=4,
    learning_rate=0.07839766598276574,
    l2_leaf_reg=9.48232922368992,
    border_count=94,
    loss_function="MultiClass",
    verbose=0,
    random_state=42
)



ensemble = VotingClassifier(
    estimators=[
        ("rf", rf_model),
        ("xgb", xgb_model),
        ("cat", cat_model)
    ],
    voting="soft"
)



# Save model
joblib.dump(ensemble, "models/ensemble_model.pkl")

# Save label encoder
joblib.dump(le, "models/label_encoder.pkl")

print(" Model saved successfully")

""" ensemble.fit(X_train, y_train)



preds = ensemble.predict(X_test)

acc = accuracy_score(y_test, preds)

print("🔥 Ensemble Accuracy:", acc)



sample_pred = ensemble.predict(X_test[:5])
decoded = le.inverse_transform(sample_pred)

print("Sample Predictions:", decoded) """