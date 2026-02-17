import joblib
from src.preprocess import load_data, build_pipeline
from src.train import train_model
from src.evaluate import evaluate_model

def main():

    path = "data/ai4i2020.csv"

    df = load_data(path)

    preprocessor, X_train, X_test, y_train, y_test = build_pipeline(df)

    model = train_model(preprocessor, X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, "industrial_failure_model.pkl")

if __name__ == "__main__":
    main()
