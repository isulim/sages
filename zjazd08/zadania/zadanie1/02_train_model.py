import pandas as pd
import mlflow
import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Random Forest")
mlflow.autolog()

X = pd.read_csv("data/ready/X_ready.csv")
y = pd.read_csv("data/ready/y_ready.csv")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    with mlflow.start_run(nested=True):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "criterion": trial.suggest_categorical("criterion", ["entropy", "gini"]),
            "max_depth": trial.suggest_int("max_depth", 1, 10, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, log=True),
            "max_features": trial.suggest_int("max_features", 1, 10, log=True),
            "random_state": 42,
        }

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        return acc


if __name__ == "__main__":

    with mlflow.start_run():
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        best_trial = study.best_trial
        print(f"Best trial (accuracy score): {best_trial.value}")
        print("Best trial (params):")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        model = RandomForestClassifier(**best_trial.params)
        model.fit(X_train, y_train)
        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(model, "model", signature=signature, registered_model_name="Random Forest")


