
import joblib
from pathlib import Path

model_dir = Path("ml_pipeline/models")

text = joblib.load(model_dir / "text_preprocessor.joblib")
structured = joblib.load(model_dir / "structured_preprocessor.joblib")

print("TEXT PREPROCESSOR:", type(text).__name__)
print("STRUCTURED PREPROCESSOR:", type(structured).__name__)

print("\nSTRUCTURED INPUT COLUMNS:")
if hasattr(structured, "feature_names_in_"):
    print(structured.feature_names_in_.tolist())
else:
    print("Not available")


print("\nTEXT SETTINGS:")
print("max_features:", text.max_features)
print("ngram_range:", text.ngram_range)
print("lowercase:", text.lowercase)
print("stop_words:", text.stop_words)
