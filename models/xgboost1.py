import pandas as pd
import optuna
import mlflow
import mlflow.xgboost

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import constants


# Load datasets
train = pd.read_csv(constants.DATA_TRAIN_PATH)
test = pd.read_csv(constants.DATA_TEST_PATH)


# Separate features and labels
X_train = train.drop(constants.TARGET_COLUMN, axis=1)
y_train = train[constants.TARGET_COLUMN]

X_test = test.drop(constants.TARGET_COLUMN, axis=1)
y_test = test[constants.TARGET_COLUMN]


# Label Encoding
le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)


mlflow.set_experiment(constants.MLFLOW_EXPERIMENT)


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
    }

    model = XGBClassifier(
        **params,
        objective="multi:softmax",
        eval_metric="mlogloss",
        use_label_encoder=False
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return acc


# Optuna study
study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=30)


best_params = study.best_params

print("\nBest Parameters:", best_params)


# Train final model
best_model = XGBClassifier(
    **best_params,
    objective="multi:softmax",
    eval_metric="mlogloss"
)

best_model.fit(X_train, y_train)

preds = best_model.predict(X_test)

acc = accuracy_score(y_test, preds)


# Log to MLflow
with mlflow.start_run(run_name="XGB_Optuna_LabelEncoded"):

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)

    mlflow.xgboost.log_model(best_model, name="xgb_optuna_model")


print("Final Accuracy:", acc)