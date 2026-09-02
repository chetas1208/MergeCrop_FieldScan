"""CropMerge Field Triage vision engine."""

__version__ = "0.1.0"

DISCLAIMER = (
    "CropMerge Field Triage analyzes visual patterns in RGB imagery. "
    "Flagged regions represent differences from surrounding field appearance "
    "and are intended for human review. They are not diagnoses of crop disease, "
    "nutrient status, irrigation failure, or plant health."
)

DEFAULT_LIMITATIONS = [
    "RGB-only visual analysis",
    "No agronomic diagnosis inferred",
    "Image-relative field map (not georeferenced unless GPS metadata present)",
    "Exploratory visual anomaly scores are not calibrated probabilities",
    "Do not manufacture NDVI from RGB",
]
