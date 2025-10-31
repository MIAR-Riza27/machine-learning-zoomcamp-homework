import pickle

# Load the pipeline
with open("pipeline_v1.bin", "rb") as f:
    model = pickle.load(f)

# Data to predict
X = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

# Predict probability
proba = model.predict_proba([X])[0, 1]
print("Probability:", round(proba, 3))