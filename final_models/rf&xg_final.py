import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier



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



rf_model = RandomForestClassifier(
    n_estimators=208,
    max_depth=14,
    min_samples_split=10,
    min_samples_leaf=1,
    max_features="log2",
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)

print(" Random Forest Accuracy:", rf_acc)


xgb_model = XGBClassifier(
    n_estimators=293,
    max_depth=4,
    learning_rate=0.0698,
    subsample=0.784,
    colsample_bytree=0.988,
    objective="multi:softmax",
    eval_metric="mlogloss",
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_preds)

print("XGBoost Accuracy:", xgb_acc)


sample_pred = xgb_model.predict(X_test[:5])
decoded = le.inverse_transform(sample_pred)

print("Sample Predictions:", decoded)