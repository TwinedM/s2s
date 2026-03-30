import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score



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



cat_model = CatBoostClassifier(
    iterations=372,
    depth=4,
    learning_rate=0.07839766598276574,
    l2_leaf_reg=9.48232922368992,
    border_count=94,
    loss_function="MultiClass",
    verbose=100,
    random_state=42
)

cat_model.fit(X_train, y_train)



cat_preds = cat_model.predict(X_test)

cat_acc = accuracy_score(y_test, cat_preds)

print(" CatBoost Final Accuracy:", cat_acc)



sample_pred = cat_model.predict(X_test[:5])
decoded = le.inverse_transform(sample_pred.astype(int).flatten())

print("Sample Predictions:", decoded)