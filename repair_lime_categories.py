
from pathlib import Path
from datetime import datetime
import shutil

import joblib
import numpy as np

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "ml_pipeline" / "models"

PREPROCESSOR_PATH = MODEL_DIR / "structured_preprocessor.joblib"
CATEGORY_VALUES_PATH = MODEL_DIR / "category_values.joblib"
CATEGORICAL_NAMES_PATH = MODEL_DIR / "categorical_names.joblib"
TRAINING_DATA_PATH = MODEL_DIR / "tabular_lime_training_data.joblib"

# Load current artifacts
structured_preprocessor = joblib.load(PREPROCESSOR_PATH)
old_category_values = joblib.load(CATEGORY_VALUES_PATH)
old_categorical_names = joblib.load(CATEGORICAL_NAMES_PATH)
training_data = np.asarray(joblib.load(TRAINING_DATA_PATH)).copy()

categorical_columns = joblib.load(
    MODEL_DIR / "categorical_columns.joblib"
)

with open(MODEL_DIR / "structured_columns.json", "r", encoding="utf-8") as f:
    structured_columns = __import__("json").load(f)

# Get the categories from the fitted, production structured preprocessor.
categorical_pipeline = structured_preprocessor.named_transformers_[
    "categorical"
]
encoder = categorical_pipeline.named_steps["encoder"]

encoder_columns = list(
    structured_preprocessor.transformers_[0][2]
)

if encoder_columns != list(categorical_columns):
    raise ValueError(
        "Categorical column order mismatch. Stop and inspect the artifacts."
    )

if training_data.ndim != 2 or training_data.shape[1] != len(structured_columns):
    raise ValueError(
        f"Unexpected LIME training-data shape: {training_data.shape}"
    )

# Back up the artifacts before changing anything.
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = MODEL_DIR / f"lime_repair_backup_{timestamp}"
backup_dir.mkdir(parents=True, exist_ok=False)

for path in [
    CATEGORY_VALUES_PATH,
    CATEGORICAL_NAMES_PATH,
    TRAINING_DATA_PATH,
]:
    shutil.copy2(path, backup_dir / path.name)

print(f"Backup created: {backup_dir}")

# Use the exact category order from the fitted encoder.
new_category_values = {}
new_categorical_names = {}

for column, categories in zip(encoder_columns, encoder.categories_):
    feature_index = structured_columns.index(column)

    # Convert labels to strings to match the existing LIME encoding.
    labels = [str(value) for value in categories]

    # The fitted encoder uses "Missing"; older LIME files used
    # "__MISSING__". Keep one consistent representation.
    labels = [
        "Missing" if value == "__MISSING__" else value
        for value in labels
    ]

    if len(labels) != len(set(labels)):
        raise ValueError(
            f"Duplicate category labels after normalization: {column}"
        )

    new_category_values[column] = labels
    new_categorical_names[feature_index] = labels

# Remap each saved training-data category from its old index to the
# matching index in the fitted encoder. Categories not present in the
# current encoder receive a dedicated unseen-category index.
repaired_data = training_data.copy()
unseen_counts = {}

for column in categorical_columns:
    feature_index = structured_columns.index(column)

    old_labels = [
        "Missing" if str(value) == "__MISSING__" else str(value)
        for value in old_category_values[column]
    ]

    new_labels = new_category_values[column]
    new_lookup = {label: i for i, label in enumerate(new_labels)}

    unseen_code = len(new_labels)
    unseen_count = 0

    for row_index, raw_code in enumerate(training_data[:, feature_index]):
        old_index = int(round(float(raw_code)))

        if old_index < 0 or old_index >= len(old_labels):
            raise ValueError(
                f"Invalid saved category index for {column}: {raw_code}"
            )

        old_label = old_labels[old_index]

        if old_label in new_lookup:
            repaired_data[row_index, feature_index] = new_lookup[old_label]
        else:
            repaired_data[row_index, feature_index] = unseen_code
            unseen_count += 1

    unseen_counts[column] = unseen_count

# Save repaired LIME artifacts only. The trained fraud model is untouched.
joblib.dump(new_category_values, CATEGORY_VALUES_PATH)
joblib.dump(new_categorical_names, CATEGORICAL_NAMES_PATH)
joblib.dump(repaired_data, TRAINING_DATA_PATH)

print("\nLIME category remapping completed.")
print("Training data shape:", repaired_data.shape)

for column, count in unseen_counts.items():
    print(f"{column}: {count} training rows mapped to __UNSEEN__")

print("\nUpdated artifacts:")
print(CATEGORY_VALUES_PATH)
print(CATEGORICAL_NAMES_PATH)
print(TRAINING_DATA_PATH)
print("\nNo classifier or model configuration was modified.")
