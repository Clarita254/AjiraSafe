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

print("UNRECOGNISED CATEGORY FREQUENCY IN LIME TRAINING DATA")
for i, (col, idx) in enumerate(zip(cat_cols, indices)):
    saved = list(cats[col])
    learned = set(encoder.categories_[i])
    unknown_rows = 0
    examples = []

    for value in training[:, idx]:
        category_index = int(round(float(value)))
        category = saved[category_index]

        # The old missing-value marker should map to the
        # current preprocessor's missing-value convention.
        if category == "__MISSING__":
            category = "Missing"

        if category not in learned:
            unknown_rows += 1
            if len(examples) < 5 and category not in examples:
                examples.append(category)

    print(
        f"\n{col}: {unknown_rows}/{len(training)} rows "
        f"({unknown_rows / len(training):.1%}) need unknown-category handling"
    )
    print("  Examples:", examples)

print("\nRead-only diagnostic complete; no files modified.")
