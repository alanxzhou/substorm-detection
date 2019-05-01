import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

results_fn = "search_results_2classes.pkl"

with open(results_fn, 'rb') as pkl:
    results = pickle.load(pkl)

best_model = sorted(results, key=lambda r: r['evaluation']['time_output_acc'])[0]
for p in best_model['params']:
    print(p, best_model['params'][p])

plt.plot(best_model['history']['val_strength_output_mean_squared_error'])
plt.show()
