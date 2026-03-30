import pandas as pd
import optuna
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import constants


# Load dataset
train = pd.read_csv(constants.DATA_TRAIN_PATH)
test = pd.read_csv(constants.DATA_TEST_PATH)

X_train = train.drop(constants.TARGET_COLUMN, axis=1)
y_train = train[constants.TARGET_COLUMN]

X_test = test.drop(constants.TARGET_COLUMN, axis=1)
y_test = test[constants.TARGET_COLUMN]


mlflow.set_experiment(constants.MLFLOW_EXPERIMENT)


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    }

    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    return acc


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=30)


best_params = study.best_params

print("\nBest Params:", best_params)


# Train final model
best_model = RandomForestClassifier(**best_params)

best_model.fit(X_train, y_train)

preds = best_model.predict(X_test)

acc = accuracy_score(y_test, preds)


with mlflow.start_run(run_name="RF_Optuna"):

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", acc)

    mlflow.sklearn.log_model(best_model, name="rf_optuna_model")

print("Final Accuracy:", acc)