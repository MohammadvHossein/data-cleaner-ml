from data_cleaner import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib


def main():
    print("=== Step 1: Load saved pipeline & model ===")
    dc = DataCleaner.load_pipeline("my_pipeline.pkl")
    model = joblib.load("model.pkl")
    print("Pipeline and model loaded successfully")

    print("\n=== Step 2: Show pipeline config ===")
    print(f"Target column: {dc.target_col}")
    print(f"Dropped columns: {dc.columns_to_drop}")
    print(f"Scalers: {list(dc.pipeline.scalers.keys())}")
    print(f"One-hot columns: {dc.pipeline.onehot_cols}")

    print("\n=== Step 3: Inference on new data ===")
    new_customers = pd.DataFrame({
        "age": [28, 42, 35],
        "salary": [65000, 95000, 78000],
        "city": ["Tehran", "Isfahan", "Shiraz"],
        "gender": ["F", "M", "F"],
    })

    processed = dc.transform(new_customers)
    predictions = model.predict(processed)
    probabilities = model.predict_proba(processed)

    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        status = "Purchased" if pred == 1 else "Not Purchased"
        print(f"  Customer {i+1}: {status} (confidence: {max(prob):.2%})")

    print("\n=== All features demonstrated ===")
    print("- load_pipeline() from disk")
    print("- transform() new raw data")
    print("- predict() / predict_proba()")
    print("- Access pipeline config (target, columns, scalers, encoders)")


if __name__ == "__main__":
    main()
