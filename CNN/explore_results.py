import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")

results_fn = "search_results_2classes.pkl"

with open(results_fn, 'rb') as pkl:
    raw_results = pickle.load(pkl, encoding='latin')

results = []
pars = raw_results[0].keys()
for model in raw_results:
    results.append({})
    for par in pars:
        results[-1][par] = model[par]

acc = [r['evaluation']['time_output_acc'] for r in results]
mae = [1/r['evaluation']['strength_output_mean_absolute_error'] for r in results]

plt.figure()
plt.plot(acc, mae, '.')

top_n = 10

sorted_results = sorted(results, key=lambda r: r['evaluation']['time_output_acc'], reverse=True)

for r in sorted_results[:top_n]:
    plt.plot(r['evaluation']['time_output_acc'], 1/r['evaluation']['strength_output_mean_absolute_error'], 'x')

s = '{!s:25}' + ''.join([' {:<10.4f}'] * top_n)
v = [''] + [r['evaluation']['time_output_acc'] for r in sorted_results[:top_n]]
print(s.format(*v))
v = [''] + [r['evaluation']['strength_output_mean_absolute_error'] for r in sorted_results[:top_n]]
print(s.format(*v))
s = '{!s:25}' + ''.join([' {!s:10}'] * top_n)
for p in results[0]['params']:
    v = [p] + [r['params'][p] for r in sorted_results[:top_n]]
    print(s.format(*v))

plt.figure()
for r in sorted_results[:top_n]:
    plt.plot(r['history']['val_strength_output_mean_squared_error'])

plt.figure()
for r in sorted_results[:top_n]:
    plt.plot(r['history']['val_time_output_acc'])
plt.show()
