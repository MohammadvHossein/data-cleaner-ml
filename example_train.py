from data_cleaner import DataCleaner
import pandas as pd
from sklearn.svm import SVC
import joblib

data = pd.DataFrame({
    "ID": range(100),
    "age": [25, 30, 35, None, 40, 45, 50, 55, 60, 65] * 10,
    "salary": [50000, 60000, None, 80000, 90000, 100000, 110000, 120000, None, 140000] * 10,
    "city": ["Tehran", "Shiraz", "Tehran", "Isfahan", None, "Tehran", "Shiraz", "Isfahan", "Tehran", "Shiraz"] * 10,
    "gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F"] * 10,
    "purchased": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 10,
})

print("=== Step 1: Load, configure & inspect ===")
dc = DataCleaner(random_state=42)
dc.load_df(data)
dc.set_target("purchased")
dc.drop_columns(["ID"])
print(dc.summary())

print("\n=== Step 2: Auto-clean with full pipeline ===")
print(f"Problem type detected: {dc._detect_problem_type(dc.df['purchased'])}")
X_train, X_test, y_train, y_test = dc.prepare(
    test_size=0.2,
    auto_scale=True,
    auto_encode=True,
    handle_nulls=True,
    auto_drop_useless=True,
    handle_outliers="clip",
    feature_engineering=False,
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

print("\n=== Step 3: View cleaned data info ===")
print(f"Features ({len(X_train.columns)}): {list(X_train.columns)}")
print(f"Scalers used: {list(dc.pipeline.scalers.keys())}")
null_strategy = "KNN" if dc.pipeline.used_knn else "drop"
print(f"Null handling: {null_strategy}")

print("\n=== Step 4: Train model ===")
model = SVC(probability=True)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

print("\n=== Step 5: Test pipeline on raw data ===")
new_data = pd.DataFrame({
    "age": [28, 42],
    "salary": [65000, 95000],
    "city": ["Tehran", "Isfahan"],
    "gender": ["F", "M"],
})
processed = dc.get_pipeline().transform(new_data)
predictions = model.predict(processed)
print(f"Predictions: {predictions}")

print("\n=== Step 6: Export cleaned dataset ===")
dc.export_cleaned("cleaned_features.csv")
dc.export_cleaned("cleaned_with_target.xlsx", include_target=True)
print("Exported: cleaned_features.csv, cleaned_with_target.xlsx")

print("\n=== Step 7: Save model & pipeline for inference ===")
joblib.dump(model, "model.pkl")
dc.save_pipeline("my_pipeline.pkl")
print("Saved: model.pkl, my_pipeline.pkl")
