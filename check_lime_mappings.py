import joblib
from pathlib import Path

p = Path("ml_pipeline/models")
cats = joblib.load(p / "category_values.joblib")
pre = joblib.load(p / "structured_preprocessor.joblib")
training = joblib.load(p / "tabular_lime_training_data.joblib")

encoder = pre.named_transformers_["categorical"].named_steps["encoder"]
cat_cols = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function"
]

print("LIME training-data shape:", training.shape)
print("Expected structured columns:", len(pre.feature_names_in_))
print("Categorical feature indices:", [list(pre.feature_names_in_).index(c) for c in cat_cols])

print("\nCATEGORY MAPPING CHECK")
for i, col in enumerate(cat_cols):
    saved = list(cats[col])
    learned = list(encoder.categories_[i])
    print(
        f"{col}: saved={len(saved)}, encoder={len(learned)}, "
        f"same_set={set(saved) == set(learned)}, "
        f"same_order={saved == learned}"
    )
