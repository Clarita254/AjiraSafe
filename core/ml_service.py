
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer

# Project-level model artifact directory
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_pipeline" / "models"

# Load the saved model artifacts
model = joblib.load(
    MODEL_DIR / "calibrated_random_forest_model.joblib"
)
text_preprocessor = joblib.load(MODEL_DIR / "text_preprocessor.joblib")
structured_preprocessor = joblib.load(
    MODEL_DIR / "structured_preprocessor.joblib"
)

with open(MODEL_DIR / "model_config.json", "r", encoding="utf-8") as file:
    model_config = json.load(file)


TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
]

STRUCTURED_COLUMNS = [
    "location",
    "department",
    "salary_range",
    "telecommuting",
    "has_company_logo",
    "has_questions",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

COMPLETENESS_COLUMNS = [
    "title",
    "location",
    "company_profile",
    "description",
    "requirements",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
]

URGENCY_KEYWORDS = [
    "urgent hiring",
    "apply immediately",
    "immediate start",
    "limited time",
    "act now",
    "start immediately",
    "hiring urgently",
]

PAYMENT_KEYWORDS = [
    "registration fee",
    "upfront payment",
    "processing fee",
    "application fee",
    "pay to apply",
    "payment required",
    "training fee",
    "deposit required",
]



# Load saved tabular LIME configuration
TABULAR_LIME_DIR = MODEL_DIR

category_values = joblib.load(
    TABULAR_LIME_DIR / "category_values.joblib"
)
categorical_names = joblib.load(
    TABULAR_LIME_DIR / "categorical_names.joblib"
)
numeric_columns = joblib.load(
    TABULAR_LIME_DIR / "numeric_columns.joblib"
)
categorical_columns = joblib.load(
    TABULAR_LIME_DIR / "categorical_columns.joblib"
)
tabular_lime_training_data = joblib.load(
    TABULAR_LIME_DIR / "tabular_lime_training_data.joblib"
)

with open(
    TABULAR_LIME_DIR / "structured_columns.json",
    "r",
    encoding="utf-8"
) as file:
    tabular_structured_columns = json.load(file)

# Confirm the saved training data matches the configuration
if tabular_lime_training_data.shape[1] != len(tabular_structured_columns):
    raise ValueError(
        "Tabular LIME training data does not match "
        "the structured column configuration."
    )


# Identify the Boolean columns.
BOOLEAN_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions",
]

# Identify the original categorical feature indices.
categorical_feature_indices = [
    tabular_structured_columns.index(column)
    for column in categorical_columns
]

# Treat Boolean features as categorical for LIME.
boolean_feature_indices = [
    tabular_structured_columns.index(column)
    for column in BOOLEAN_COLUMNS
]

# Combine and sort all categorical feature indices.
lime_categorical_feature_indices = sorted(
    categorical_feature_indices + boolean_feature_indices
)

# Give each categorical feature an explicit label for unseen values.
lime_categorical_names = {
    index: list(categorical_names[index]) + ["__UNSEEN__"]
    for index in categorical_feature_indices
}

# Provide category names for Boolean features.
for index in boolean_feature_indices:
    lime_categorical_names[index] = ["No", "Yes"]

# Reconstruct the experimental LIME explainer.
tabular_explainer = LimeTabularExplainer(
    training_data=np.asarray(tabular_lime_training_data),
    feature_names=tabular_structured_columns,
    categorical_features=lime_categorical_feature_indices,
    categorical_names=lime_categorical_names,
    class_names=["Legitimate", "Fraudulent"],
    mode="classification",
    discretize_continuous=False,
    sample_around_instance=False,
    random_state=42,
)

def normalize_boolean(value):
    """Normalize common HTML form, string, numeric and Boolean values."""
    if value is None or pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, float, np.integer, np.floating)):
        if value == 1:
            return True
        if value == 0:
            return False
        raise ValueError(f"Invalid Boolean value: {value!r}")

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "on"}:
        return True

    if normalized in {"false", "0", "no", "off", ""}:
        return False

    raise ValueError(f"Invalid Boolean value: {value!r}")


def normalize_boolean_columns(dataframe):
    """Apply the same Boolean conversion to all model input paths."""
    dataframe = dataframe.copy()

    for column in [
        "telecommuting",
        "has_company_logo",
        "has_questions",
    ]:
        dataframe[column] = dataframe[column].map(normalize_boolean)

    return dataframe
def normalize_categorical_value(value, column):
    """
    Standardise categorical inputs against the categories
    learned by the fitted structured preprocessor.

    Known equivalent values are mapped to their canonical
    training category. Unknown values remain unknown.
    """

    # Preserve missing values for the fitted SimpleImputer.
    if value is None or pd.isna(value):
        return np.nan

    value = " ".join(str(value).split())

    if not value:
        return np.nan

    # Explicit aliases for genuinely equivalent education values.
    aliases = {
        "required_education": {
            "bachelor degree": "Bachelor's Degree",
            "bachelors degree": "Bachelor's Degree",
            "master degree": "Master's Degree",
            "masters degree": "Master's Degree",
        }
    }

    normalised_key = value.casefold()

    # Apply an explicit alias if one exists.
    column_aliases = aliases.get(column, {})
    if normalised_key in column_aliases:
        return column_aliases[normalised_key]

    # Match against the exact categories learned during training.
    categories = category_values[column]

    category_lookup = {
        str(category).casefold(): category
        for category in categories
    }

    if normalised_key in category_lookup:
        return category_lookup[normalised_key]

    # Do not invent a category for an unfamiliar input.
    return value
def count_keyword_matches(text, keywords):
    """Count keyword phrases using the notebook's matching rule."""
    text = str(text).lower()
    return sum(
        len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
        for keyword in keywords
    )


def prepare_job_features(job_data):
    """Prepare model inputs for one job posting."""

    # Create a single-row DataFrame with all required source columns.
    data = {
        column: job_data.get(column, "")
        for column in TEXT_COLUMNS + STRUCTURED_COLUMNS
    }
    job_df = pd.DataFrame([data])

    # Match the notebook's text preparation.
    text_df = job_df[TEXT_COLUMNS].copy()
    text_df = text_df.fillna("")

    combined_text = text_df.agg(" ".join, axis=1)
    combined_text = combined_text.str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    # Match the notebook's missing-value behavior:
    # blank strings become missing values for completeness scoring.
    completeness_df = job_df[COMPLETENESS_COLUMNS].copy()
    completeness_df = completeness_df.replace(
        r"^\s*$", float("nan"), regex=True
    )

    # Calculate engineered features.
    engineered = pd.DataFrame(index=job_df.index)
    engineered["urgency_score"] = combined_text.map(
        lambda text: count_keyword_matches(text, URGENCY_KEYWORDS)
    )
    engineered["payment_keyword_score"] = combined_text.map(
        lambda text: count_keyword_matches(text, PAYMENT_KEYWORDS)
    )
    engineered["missing_information_score"] = (
        completeness_df.isna().mean(axis=1)
    )

    # Convert structured Boolean fields into booleans.
          # Normalise categorical fields before model preprocessing.
    for column in categorical_columns:
        job_df[column] = job_df[column].map(
            lambda value: normalize_categorical_value(value, column)
        )

    # Apply consistent Boolean normalisation.
    job_df = normalize_boolean_columns(job_df)

    # Transform using the saved preprocessors.
    text_features = text_preprocessor.transform(combined_text)
    structured_features = structured_preprocessor.transform(
        job_df[STRUCTURED_COLUMNS]
    )

   
    # Keep the feature order: text, structured, engineered.
    combined_features = hstack(
        [
            text_features,
            csr_matrix(structured_features),
            csr_matrix(engineered.to_numpy()),
        ],
        format="csr",
    )

    expected_features = model_config["total_model_features"]
    if combined_features.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features, "
            f"but generated {combined_features.shape[1]}."
        )

    return combined_features




def assess_job(job_data):
    """Return a model prediction and risk score for one job."""

    features = prepare_job_features(job_data)

    # Obtain the calibrated fraud probability.
    fraud_probability = float(
        model.predict_proba(features)[0][1]
    )

    # Convert probability into a risk score.
    risk_score = round(fraud_probability * 100, 2)

    # Apply the selected classification threshold.
    threshold = float(model_config["classification_threshold"])
    predicted_fraud = fraud_probability >= threshold

    # Determine risk level using the selected classification boundary.
    # Use the unrounded probability to avoid rounding inconsistencies.
    if fraud_probability < threshold:
        risk_level = "low"
    elif fraud_probability < 0.50:
        risk_level = "medium"
    elif fraud_probability < 0.75:
        risk_level = "high"
    else:
        risk_level = "very_high"

    return {
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "prediction": (
            "fraudulent" if predicted_fraud else "legitimate"
        ),
        "threshold": threshold,
    }



def lime_predict_proba(texts, structured_row, engineered_row):
    """
    Return model probabilities for LIME-perturbed text.

    Structured fields and the original missing-information score
    remain fixed. Urgency and payment keyword scores are recalculated
    for each perturbed text sample.
    """

    texts = list(texts)

    if not texts:
        return np.empty((0, 2))

    text_series = pd.Series(texts, dtype=str)

    # Transform perturbed text samples.
    text_matrix = text_preprocessor.transform(text_series)

    # Repeat the original structured fields.
    structured_batch = pd.DataFrame(
        [structured_row.to_dict() for _ in texts],
        columns=STRUCTURED_COLUMNS,
    )

    structured_matrix = structured_preprocessor.transform(
        structured_batch
    )

    # Recalculate keyword scores for every perturbed text.
    urgency_scores = [
        count_keyword_matches(text, URGENCY_KEYWORDS)
        for text in texts
    ]

    payment_scores = [
        count_keyword_matches(text, PAYMENT_KEYWORDS)
        for text in texts
    ]

    # Keep the original missing-information score fixed.
    missing_score = float(
        engineered_row["missing_information_score"]
    )

    engineered_matrix = csr_matrix(
        np.column_stack([
            urgency_scores,
            payment_scores,
            np.repeat(missing_score, len(texts)),
        ]).astype(float)
    )

    combined_matrix = hstack(
        [
            text_matrix,
            csr_matrix(structured_matrix),
            engineered_matrix,
        ],
        format="csr",
    )

    expected_features = model_config["total_model_features"]

    if combined_matrix.shape[1] != expected_features:
        raise ValueError(
            f"LIME generated {combined_matrix.shape[1]} features; "
            f"expected {expected_features}."
        )

    return model.predict_proba(combined_matrix)

def explain_job_lime(job_data, num_features=12, num_samples=1000):
    """
    Generate a genuine LIME text explanation for one job posting.

    Positive weights support the local explanation for the fraudulent
    class; negative weights support the legitimate class.
    """
    # Build the same input data used by assess_job().
    data = {
        column: job_data.get(column, "")
        for column in TEXT_COLUMNS + STRUCTURED_COLUMNS
    }
    job_df = pd.DataFrame([data])

    # Reconstruct the combined text exactly as in prepare_job_features().
    text_df = job_df[TEXT_COLUMNS].copy().fillna("")
    combined_text = text_df.agg(" ".join, axis=1)
    combined_text = combined_text.str.replace(
        r"\s+", " ", regex=True
    ).str.strip()

    original_text = str(combined_text.iloc[0])

    # Prepare the original structured row.
    structured_row = job_df[STRUCTURED_COLUMNS].copy()

    # Apply the same categorical normalisation as the main prediction.
    for column in categorical_columns:
        structured_row[column] = structured_row[column].map(
            lambda value: normalize_categorical_value(value, column)
        )

    # Apply the same Boolean normalisation as the main prediction.
    structured_row = normalize_boolean_columns(structured_row)

    structured_row = structured_row.iloc[0]
    # Calculate the original missing-information score.
    completeness_df = job_df[COMPLETENESS_COLUMNS].copy()
    completeness_df = completeness_df.replace(
        r"^\s*$", float("nan"), regex=True
    )
    missing_score = float(completeness_df.isna().mean(axis=1).iloc[0])

    engineered_row = pd.Series({
        "missing_information_score": missing_score
    })

    # Generate the local explanation for the fraudulent class (1).
    explainer = LimeTextExplainer(
        class_names=["Legitimate", "Fraudulent"],
        random_state=42,
    )

    explanation = explainer.explain_instance(
        text_instance=original_text,
        classifier_fn=lambda texts: lime_predict_proba(
            texts, structured_row, engineered_row
        ),
        labels=(1,),
        num_features=num_features,
        num_samples=num_samples,
    )

    # Store the actual LIME feature weights in a serializable format.
    feature_weights = [
    {
        "feature": str(feature),
        "weight": round(float(weight), 6),
    }
    for feature, weight in explanation.as_list(label=1)
]
    positive_features = [
        item["feature"]
        for item in feature_weights
        if item["weight"] > 0
    ]
    negative_features = [
        item["feature"]
        for item in feature_weights
        if item["weight"] < 0
    ]

    return {
        "method": "LIME text explanation",
        "features": feature_weights,
        "positive_features": positive_features,
        "negative_features": negative_features,
        "num_features": num_features,
        "num_samples": num_samples,
    }



def encode_structured_row(structured_row):
    """Encode one structured row using the saved LIME mappings."""
    encoded = []

    for column in tabular_structured_columns:
        value = structured_row[column]

        if column in categorical_columns:
            categories = category_values[column]

            if pd.isna(value) or str(value).strip() == "":
                value = "Missing"
            else:
                value = str(value)

            try:
                code = categories.index(value)
            except ValueError:
                # Use a separate code for unseen categories.
                code = len(categories)

            encoded.append(code)

        else:
            numeric_value = pd.to_numeric(value, errors="coerce")
            encoded.append(
                0.0 if pd.isna(numeric_value) else float(numeric_value)
            )

    return np.asarray(encoded, dtype=float)



def decode_structured_rows(encoded_rows, original_job_data=None):
    """Decode LIME rows while preserving unknown and missing categories."""
    decoded = []

    for encoded_row in np.asarray(encoded_rows):
        row = {}

        for index, column in enumerate(tabular_structured_columns):
            value = encoded_row[index]

            if column in categorical_columns:
                categories = category_values[column]
                category_index = int(round(float(value)))

                if category_index >= len(categories):
                    original_value = (
                        original_job_data.get(column)
                        if original_job_data is not None else None
                    )

                    original_is_missing = (
                        original_value is None
                        or pd.isna(original_value)
                        or str(original_value).strip() == ""
                    )

                    original_is_unseen = (
                        not original_is_missing
                        and str(original_value) not in categories
                    )

                    if original_is_unseen:
                        row[column] = str(original_value)
                    else:
                        row[column] = "__UNSEEN__"

                elif category_index < 0:
                    row[column] = np.nan

                else:
                    category = categories[category_index]

                    row[column] = (
                        np.nan
                        if category in ("__MISSING__", "Missing")
                        else category
                    )

            else:
                row[column] = float(value)

        decoded.append(row)

    return pd.DataFrame(decoded, columns=tabular_structured_columns)


def tabular_lime_predict_proba(encoded_rows, job_data):
    """
    Predict fraud probabilities for LIME-perturbed structured fields.
    Text remains fixed; text-dependent engineered scores are recalculated.
    """

    decoded_rows = decode_structured_rows(
        encoded_rows,
        original_job_data=job_data
    )

    # Reconstruct the original combined text in the same way as
    # prepare_job_features().
    text_values = {
        column: job_data.get(column, "")
        for column in TEXT_COLUMNS
    }

    text_df = pd.DataFrame([text_values]).fillna("")

    original_text = (
        text_df.agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .iloc[0]
    )

    # Apply the same categorical normalisation as the main prediction.
    for column in categorical_columns:
        decoded_rows[column] = decoded_rows[column].map(
            lambda value: normalize_categorical_value(value, column)
        )

    
    # Convert LIME's perturbed numerical Boolean features back to
    # Boolean values using the original 0/1 representation.
    for column in ["telecommuting", "has_company_logo", "has_questions"]:
        decoded_rows[column] = (
            pd.to_numeric(decoded_rows[column], errors="coerce")
            .fillna(0)
            .ge(0.5)
        )

    # Transform fixed text for every perturbed structured row.
    text_matrix = text_preprocessor.transform(
        pd.Series([original_text] * len(decoded_rows), dtype=str)
    )

    structured_matrix = structured_preprocessor.transform(
        decoded_rows[STRUCTURED_COLUMNS]
    )

    # Recalculate text-dependent engineered features.
    urgency_score = count_keyword_matches(
        original_text, URGENCY_KEYWORDS
    )

    payment_score = count_keyword_matches(
        original_text, PAYMENT_KEYWORDS
    )

  
    # Recalculate missing-information scores for each
    # LIME-perturbed structured row.
    completeness_batch = pd.DataFrame(
        {
            column: (
                decoded_rows[column]
                if column in decoded_rows.columns
                else [job_data.get(column, "")] * len(decoded_rows)
            )
            for column in COMPLETENESS_COLUMNS
        }
    )

    # Match the missing-value treatment in prepare_job_features().
    completeness_batch = completeness_batch.replace(
        r"^\s*$", np.nan, regex=True
    )

    missing_scores = (
        completeness_batch.isna()
        .mean(axis=1)
        .to_numpy()
    )

    engineered_matrix = csr_matrix(
        np.column_stack([
            np.repeat(urgency_score, len(decoded_rows)),
            np.repeat(payment_score, len(decoded_rows)),
            missing_scores,
        ]).astype(float)
    )

    combined_matrix = hstack(
        [
            text_matrix,
            csr_matrix(structured_matrix),
            engineered_matrix,
        ],
        format="csr",
    )

    if combined_matrix.shape[1] != model_config["total_model_features"]:
        raise ValueError(
            "Tabular LIME produced an unexpected number of model features."
        )

    return model.predict_proba(combined_matrix)


def explain_job_tabular_lime(job_data, num_features=11, num_samples=1000):
    """Explain how structured fields locally influence the fraud score."""

    structured_values = {
        column: job_data.get(column, "")
        for column in STRUCTURED_COLUMNS
    }

    structured_row = pd.DataFrame([structured_values])

    # Apply the same categorical normalisation as the main prediction.
    for column in categorical_columns:
        structured_row[column] = structured_row[column].map(
            lambda value: normalize_categorical_value(value, column)
        )

    # Apply the same Boolean normalisation as the main prediction.
    structured_row = normalize_boolean_columns(structured_row)

    # Encode the normalised structured row for LIME.
    encoded_row = encode_structured_row(structured_row.iloc[0])

    # Reset LIME's random state to ensure reproducible explanations.
    tabular_explainer.random_state = np.random.RandomState(42)

    explanation = tabular_explainer.explain_instance(
        data_row=encoded_row,
        predict_fn=lambda rows: tabular_lime_predict_proba(rows, job_data),
        labels=(1,),
        num_features=num_features,
        num_samples=num_samples,
    )

    features = []

    for feature, weight in explanation.as_list(label=1):
        feature_label = str(feature)

        # Replace the generic unseen-category label only when the
        # normalised original value is genuinely unknown.
        for column in categorical_columns:
            original_value = job_data.get(column)

            if (
                original_value is not None
                and not pd.isna(original_value)
                and str(original_value).strip() != ""
            ):
                normalized_value = normalize_categorical_value(
                    original_value, column
                )

                if normalized_value not in category_values[column]:
                    unseen_label = f"{column}=__UNSEEN__"

                    if feature_label == unseen_label:
                        feature_label = f"{column}={original_value}"

        features.append({
            "feature": feature_label,
            "weight": round(float(weight), 6),
        })


    # Obtain the original Random Forest prediction.
    model_features = prepare_job_features(job_data)

    model_prediction = float(
        model.predict_proba(model_features)[0, 1]
    )

    # Verify that the encoded structured row, after LIME decoding
    # and preprocessing, reproduces the original model prediction.
    # This checks that LIME is explaining the same input as assess_job().
    lime_input_probabilities = tabular_lime_predict_proba(
        encoded_row.reshape(1, -1),
        job_data
    )

    lime_input_prediction = float(
        lime_input_probabilities[0, 1]
    )

    input_prediction_difference = abs(
        model_prediction - lime_input_prediction
    )

    input_prediction_matches_model = bool(
        np.isclose(
            model_prediction,
            lime_input_prediction,
            rtol=1e-5,
            atol=1e-6
        )
    )

    if not input_prediction_matches_model:
        raise ValueError(
            "Structured LIME input does not reproduce the original "
            "Random Forest prediction. Check categorical encoding, "
            "missing-value handling and Boolean preprocessing. "
            f"Model probability: {model_prediction:.8f}; "
            f"LIME input probability: {lime_input_prediction:.8f}."
        )

    # Extract the local surrogate output.
    # IMPORTANT: This is an unbounded regression output, NOT a probability.
    surrogate_output = float(
        np.asarray(explanation.local_pred).reshape(-1)[0]
    )

    # LIME's score is the weighted R-squared of the local surrogate.
    # It measures local approximation quality, not model accuracy
    # or confidence in the fraud prediction.
    local_fidelity = float(explanation.score)

    return {
        "method": "LIME structured-feature explanation",
        "features": features,

        "positive_features": [
            item for item in features
            if item["weight"] > 0
        ],

        "negative_features": [
            item for item in features
            if item["weight"] < 0
        ],

        "num_features": num_features,
        "num_samples": num_samples,

        # LIME explanation diagnostics.
        "local_fidelity": local_fidelity,
        "model_prediction": model_prediction,

        # Clearly identified as a surrogate output, not a probability.
        "surrogate_output": surrogate_output,
        "surrogate_output_is_probability": False,

        # Input consistency diagnostics.
        "lime_input_prediction": lime_input_prediction,
        "input_prediction_difference": input_prediction_difference,
        "input_prediction_matches_model": input_prediction_matches_model,
    }
