import numpy as np
from sklearn.neural_network import MLPClassifier
import pyswarms as ps

def run_pso(x_train_scaled, y_train, x_test_scaled, y_test):

    def objective_function(params):
        n_particles = params.shape[0]
        scores = []

        for i in range(n_particles):
            hidden_size = int(params[i][0])
            lr = params[i][1]

            model = MLPClassifier(hidden_layer_sizes=(hidden_size,),
                                  learning_rate_init=lr,
                                  max_iter=500,
                                  random_state=42)

            model.fit(x_train_scaled, y_train)
            acc = model.score(x_test_scaled, y_test)

            scores.append(-acc)

        return np.array(scores)

    bounds = (np.array([50, 0.001]), np.array([200, 0.05]))

    optimizer = ps.single.GlobalBestPSO(
        n_particles=15,
        dimensions=2,
        options={'c1': 1.5, 'c2': 1.5, 'w': 0.7},
        bounds=bounds
    )

    cost, best_params = optimizer.optimize(objective_function, iters=20)

    return best_params, optimizer.cost_history