import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "iris.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.joblib"

def main():
    df = pd.read_csv(DATA_PATH)
    # Features and target
    X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
    y = df["species_name"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print("Model saved to", MODEL_PATH)

if __name__ == "__main__":
    main()
