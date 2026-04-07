import joblib
import os
from sklearn.neural_network import MLPClassifier
from pso import run_pso


def run_controller(MODE, x_train_scaled, y_train, x_test_scaled, y_test):

    model_path = "pso_model.pkl"
    param_path = "pso_params.pkl"

    print(f"MODE SELECTED: {MODE}")


    if MODE == 1:
        if os.path.exists(model_path) and os.path.exists(param_path):
            print("Loading saved model")
            mlp_pso = joblib.load(model_path)
            best_params = joblib.load(param_path)
            return mlp_pso, best_params, None
        else:
            print("No saved model found. Switching to training")

    
    if MODE == 3:
        print("deleting old model")

        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(param_path):
            os.remove(param_path)

    if MODE == 2:
        print("Running PSO Optimization")

    best_params, cost_history = run_pso(
        x_train_scaled, y_train, x_test_scaled, y_test
    )

    best_hidden = int(best_params[0])
    best_lr = best_params[1]

    mlp_pso = MLPClassifier(
        hidden_layer_sizes=(best_hidden,),
        learning_rate_init=best_lr,
        max_iter=800,
        random_state=42
    )

    mlp_pso.fit(x_train_scaled, y_train)

    joblib.dump(mlp_pso, model_path)
    joblib.dump(best_params, param_path)

    print("Model saved successfully")

    return mlp_pso, best_params, cost_history