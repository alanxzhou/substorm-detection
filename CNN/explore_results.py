import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

results_fn = "search_results_3.pkl"

with open(results_fn, 'rb') as pkl:
    results = pickle.load(pkl)

acc = [r['evaluation']['time_output_acc'] for r in results]
mae = [1/r['evaluation']['strength_output_mean_absolute_error'] for r in results]

plt.figure()
plt.plot(acc, mae, '.')

for m in sorted(results, key=lambda r: r['evaluation']['time_output_acc'], reverse=True)[:5]:
    print(results.index(m), m['evaluation']['time_output_acc'], 1/m['evaluation']['strength_output_mean_absolute_error'])

best_model = sorted(results, key=lambda r: r['evaluation']['time_output_acc'], reverse=True)[4]
plt.plot(best_model['evaluation']['time_output_acc'], 1/best_model['evaluation']['strength_output_mean_absolute_error'], 'x')
for p in best_model['params']:
    s = "{!s:25} {!s:10} {!s:10} {!s:10} {!s:10} {!s:10}".format(p, results[20]['params'][p], results[7]['params'][p],
                                                results[21]['params'][p], results[36]['params'][p],
                                                results[16]['params'][p])
    print(s)

# plt.figure()
# plt.plot(best_model['history']['val_strength_output_mean_squared_error'])
plt.show()
