import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
def glucose_simulation(model, scaler, X):
    glucose_range = np.linspace(50, 200, 100)
    base_sample = X.mean().values.reshape(1, -1)
    glucose_index = list(X.columns).index("Glucose")
    probabilities = []
    for g in glucose_range:
        temp = base_sample.copy()
        temp[0][glucose_index] = g
        temp_scaled = scaler.transform(temp)
        prob = model.predict_proba(temp_scaled)[0][1]
        probabilities.append(prob)
    plt.figure(figsize=(10,6))
    plt.plot(glucose_range, probabilities, linewidth=2)
    plt.title("Glucose vs Diabetes Probability")
    plt.xlabel("Glucose Level")
    plt.ylabel("Probability of Diabetes")
    plt.grid()
    plt.show()