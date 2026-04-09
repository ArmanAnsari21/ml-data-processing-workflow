import json
import os

from sklearn.model_selection import train_test_split

from data_loader import load_data
from preprocessing import preprocess_data
from model_training import train_model
from evaluation import evaluate_model


DATA_FILE = "sample_data/dataset.csv"
OUTPUT_FILE = "output/model_results.json"


def run_pipeline():
    df = load_data(DATA_FILE)

    df = preprocess_data(df)

    X = df[["feature1", "feature2", "feature_sum"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    model = train_model(X_train, y_train)

    accuracy = evaluate_model(model, X_test, y_test)

    os.makedirs("output", exist_ok=True)

    results = {
        "model": "LogisticRegression",
        "accuracy": accuracy
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(results, file, indent=2)

    print("ML pipeline completed.")
    print(results)


if __name__ == "__main__":
    run_pipeline()
