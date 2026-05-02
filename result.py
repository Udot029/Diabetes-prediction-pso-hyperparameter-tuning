import matplotlib.pyplot as plt
def plot_graph(baseline, pso, lstm):
    models = ['Baseline MLP', 'PSO-MLP', 'LSTM']
    accuracies = [baseline, pso, lstm]
    plt.bar(models, accuracies)
    plt.show()