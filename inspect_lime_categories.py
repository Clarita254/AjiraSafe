import joblib
import numpy as np
from pathlib import Path

p = Path("ml_pipeline/models")
cats = joblib.load(p / "category_values.joblib")
training = np.asarray(joblib.load(p / "tabular_lime_training_data.joblib"))
pre = joblib.load(p / "structured_preprocessor.joblib")
encoder = pre.named_transformers_["categorical"].named_steps["encoder"]

cat_cols = [
    "location", "department", "salary_range", "employment_type",
    "required_experience", "required_education", "industry", "function"
]
indices = [list(pre.feature_names_in_).index(c) for c in cat_cols]

print("CATEGORY DIFFERENCES")
for i, col in enumerate(cat_cols):
    saved = list(cats[col])
    learned = list(encoder.categories_[i])
    saved_set, learned_set = set(saved), set(learned)

    print(f"\n{col}")
    print("  Only in saved mapping:", len(saved_set - learned_set))
    print("  Only in encoder:", len(learned_set - saved_set))
    print("  Saved-only examples:", list(saved_set - learned_set)[:5])
    print("  Encoder-only examples:", list(learned_set - saved_set)[:5])

print("\nLIME TRAINING-DATA INDEX CHECK")
print("Shape:", training.shape)
for col, idx in zip(cat_cols, indices):
    values = training[:, idx].astype(float)
    categories = list(cats[col])
    valid = np.isfinite(values) & (values >= 0) & (values < len(categories))
    print(
        f"{col}: min={np.nanmin(values)}, max={np.nanmax(values)}, "
        f"valid_indices={valid.sum()}/{len(values)}, "
        f"saved_category_count={len(categories)}"
    )

print("\nNo files were modified.")
