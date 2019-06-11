import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

results_fn = "search_results_2classes.pkl"

with open(results_fn, 'rb') as pkl:
    raw_results = pickle.load(pkl, encoding='latin')

results = []
pars = raw_results[0].keys()
for model in raw_results:
    results.append({})
    for par in pars:
        results[-1][par] = model[par]

models = []
for r in results:
    model = {}
    for p in r['params']:
        if isinstance(r['params'][p], tuple):
            for i in range(len(r['params'][p])):
                model[p+str(i)] = r['params'][p][i]
        else:
            model[p] = r['params'][p]
    model['mag_conv_layers'] = r['params']['mag_stages'] * r['params']['mag_blocks_per_stage']
    model['sw_conv_layers'] = r['params']['sw_stages'] * r['params']['sw_blocks_per_stage']
    model['accuracy'] = r['evaluation']['time_output_acc']
    model['mae'] = r['evaluation']['strength_output_mean_absolute_error']
    models.append(model)

df = pd.DataFrame(models)
print(df.shape)

print(df.sort_values(by='accuracy').iloc[:10])

for p in df:
    if isinstance(df[p][0], tuple):
        continue
    if p == 'time_output_weight':
        fig, ax = plt.subplots()
        ax.set_ylabel("accuracy")
        ax.set_xlabel(p)
        ax.scatter(df[p] * (1 + np.random.rand(df[p].shape[0])), df['accuracy'])
        ax.set_xscale('log')
    elif len(np.unique(df[p].values)) <= 12:
        g = sns.catplot(p, 'accuracy', data=df, kind='point')
    else:
        fig, ax = plt.subplots()
        ax.set_ylabel("accuracy")
        ax.set_xlabel(p)
        ax.scatter(df[p], df['accuracy'])
    plt.savefig("C:\\Users\\Greg\\Desktop\\plots\\hps\\" + p + ".png")
    plt.close(plt.gcf())

# acc = [r['evaluation']['time_output_acc'] for r in results]
# mae = [1/r['evaluation']['strength_output_mean_absolute_error'] for r in results]
#
# plt.figure()
# plt.plot(acc, mae, '.')
#
# top_n = 10
#
# sorted_results = sorted(results, key=lambda r: r['evaluation']['time_output_acc'], reverse=True)
#
# for r in sorted_results[:top_n]:
#     plt.plot(r['evaluation']['time_output_acc'], 1/r['evaluation']['strength_output_mean_absolute_error'], 'x')
#
# s = '{!s:25}' + ''.join([' {:<10.4f}'] * top_n)
# v = ['accuracy'] + [r['evaluation']['time_output_acc'] for r in sorted_results[:top_n]]
# print(s.format(*v))
# v = ['strength mae'] + [r['evaluation']['strength_output_mean_absolute_error'] for r in sorted_results[:top_n]]
# print(s.format(*v))
# v = ['mag conv layers'] + [int(r['params']['mag_blocks_per_stage'] * r['params']['mag_stages']) for r in sorted_results[:top_n]]
# print(s.format(*v))
# v = ['sw conv layers'] + [int(r['params']['sw_blocks_per_stage'] * r['params']['sw_stages']) for r in sorted_results[:top_n]]
# print(s.format(*v))
# s = '{!s:25}' + ''.join([' {!s:10}'] * top_n)
# for p in results[0]['params']:
#     v = [p] + [r['params'][p] for r in sorted_results[:top_n]]
#     print(s.format(*v))
#
# plt.figure()
# for r in sorted_results[:top_n]:
#     plt.plot(r['history']['val_strength_output_mean_squared_error'])
#
# plt.figure()
# for r in sorted_results[:top_n]:
#     plt.plot(r['history']['val_time_output_acc'])
# plt.show()
