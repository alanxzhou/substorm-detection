import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

results_fn = "search_results_3class.pkl"

with open(results_fn, 'rb') as pkl:
    results = pickle.load(pkl)

for u in sorted(results, key=lambda r: r['confusion_matrix'][2, 1] + r['confusion_matrix'][1, 2])[:3]:
    print(u['params'])
    print(u['confusion_matrix'])
    print()

