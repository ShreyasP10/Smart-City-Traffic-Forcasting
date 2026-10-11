import joblib, numpy as np
bundle = joblib.load("base_model.pkl")
model, thresh = bundle["model"], bundle["thresholds"]

def classify(vol):
    if vol <= thresh[0]:
        return "Low"
    if vol <= thresh[1]:
        return "Moderate"
    if vol <= thresh[2]:
        return "High"
    return "Severe"

# Test a rush-hour weekday
sample = np.array([[8, 0, 6, 0, 0, 290, 0, 0, 40, 0]])  # 8am Monday
print("Predicted:", model.predict(sample)[0], "->", classify(model.predict(sample)[0]))
