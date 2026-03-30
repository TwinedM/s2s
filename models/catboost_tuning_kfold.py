import pandas as pd
import optuna
import numpy as np
import mlflow
import mlflow.catboost

from catboost import CatBoostClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder



train_df = pd.read_csv("dataset/master_alphabets_train.csv")
test_df = pd.read_csv("dataset/master_alphabets_test.csv")

TARGET_COLUMN = "label"

X = train_df.drop(TARGET_COLUMN, axis=1)
y = train_df[TARGET_COLUMN]

X_test = test_df.drop(TARGET_COLUMN, axis=1)
y_test = test_df[TARGET_COLUMN]



le = LabelEncoder()
y = le.fit_transform(y)
y_test = le.transform(y_test)



kf = KFold(n_splits=5, shuffle=True, random_state=42)



mlflow.set_experiment("CatBoost_Optuna_KFold")



def objective(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 200, 400),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "border_count": trial.suggest_int("border_count", 32, 128),
        "loss_function": "MultiClass",
        "verbose": 0
    }

    fold_accuracies = []

    # Nested MLflow run for each trial
    with mlflow.start_run(nested=True):

        mlflow.log_params(params)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            fold_accuracies.append(acc)

            # 🔹 Terminal output
            print(f"Trial {trial.number} | Fold {fold+1} Accuracy: {acc:.4f}")

            # 🔹 MLflow logging per fold
            mlflow.log_metric(f"fold_{fold+1}_accuracy", acc)

        mean_acc = np.mean(fold_accuracies)

        print(f"Trial {trial.number} | Mean CV Accuracy: {mean_acc:.4f}")
        print("-" * 50)

        mlflow.log_metric("mean_cv_accuracy", mean_acc)

    return mean_acc



study = optuna.create_study(direction="maximize")

# Main MLflow run
with mlflow.start_run(run_name="CatBoost_Optuna_KFold_Main"):

    study.optimize(objective, n_trials=10)

    best_params = study.best_params

    print("\n Best Parameters:", best_params)

    mlflow.log_params(best_params)



final_model = CatBoostClassifier(
    **best_params,
    loss_function="MultiClass",
    verbose=100
)

final_model.fit(X, y)



test_preds = final_model.predict(X_test)

test_acc = accuracy_score(y_test, test_preds)

print("\n✅ Final Test Accuracy:", test_acc)



with mlflow.start_run(run_name="CatBoost_Final_Model"):

    mlflow.log_params(best_params)
    mlflow.log_metric("test_accuracy", test_acc)

    mlflow.catboost.log_model(final_model, name="catboost_model")