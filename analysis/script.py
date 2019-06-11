import matplotlib.pyplot as plt
from analysis.visualizations import Visualizer
from sklearn.cluster import AgglomerativeClustering
import keras
import numpy as np
from analysis import nn_vis

TRAIN = False

data_fn = "../data/2classes_data128.npz"
train_val_split = .15
model_file = "../CNN/saved models/final_cnn_model.h5"

params = {'batch_size': 8, 'epochs': 18, 'verbose': 2, 'n_classes': 2,
          'time_output_weight': 1000000, 'SW': True,

          'Tm': 96, 'mag_stages': 1, 'mag_blocks_per_stage': 4,
          'mag_downsampling_strides': (2, 3),
          'mag_kernel_size': (2, 11), 'mag_fl_filters': 16,
          'mag_fl_strides': (1, 3),
          'mag_fl_kernel_size': (1, 13), 'mag_type': 'basic',

          'Tw': 192, 'sw_stages': 1, 'sw_blocks_per_stage': 1,
          'sw_downsampling_strides': 4, 'sw_kernel_size': 7, 'sw_fl_filters': 16,
          'sw_fl_strides': 3, 'sw_fl_kernel_size': 15, 'sw_type': 'residual'}

visualizer = Visualizer(data_fn, params, train_model=TRAIN, train_val_split=train_val_split, model_file=model_file)

dense_weights = visualizer.model.get_layer('time_output').get_weights()[0][-32:, 0]

relu_layers = [l for l in visualizer.model.layers if isinstance(l, keras.layers.ReLU)]
select_layer = 7
layer = relu_layers[select_layer]
activations = visualizer.get_layer_output(layer, visualizer.test_data)
cov = np.corrcoef(activations.reshape((-1, 32)).T)
i = -1
clust = AgglomerativeClustering(n_clusters=5, linkage="average", affinity="precomputed")
clust.fit(np.exp(-1 * cov))
clusters = np.argsort(clust.labels_)

fig = plt.figure(figsize=(12, 12))
plt.pcolormesh(cov[clusters][:, clusters], cmap='coolwarm')
plt.xticks(np.arange(32), clusters)
plt.yticks(np.arange(32), clusters)
plt.savefig("C:\\Users\\Greg\\Desktop\\semeter_meeting\\correlation.png")
plt.clf()
plt.close(fig)

# for each filter:
#   optimize for highly activating input (feature visualization)
#   get examples (15 or so) of highly activating input data
#   for 2 or 3 different neurons, get examples of highly activating inputs, look to see what features are translated

steps = 2000

feature_vis = nn_vis.feature_visualization(layer.output[:, 20, 2, 4], visualizer.model.inputs[0], steps)
plt.figure()
plt.pcolormesh(feature_vis[0, :, :, 0], cmap='gray')
feature_vis = nn_vis.feature_visualization(layer.output[:, 20, 5, 4], visualizer.model.inputs[0], steps)
plt.figure()
plt.pcolormesh(feature_vis[0, :, :, 0], cmap='gray')
feature_vis = nn_vis.feature_visualization(layer.output[:, 20, 8, 4], visualizer.model.inputs[0], steps)
plt.figure()
plt.pcolormesh(feature_vis[0, :, :, 0], cmap='gray')
plt.show()


# for filt in range(activations.shape[-1]):
#     print(filt)
#     max_vals = activations[:, :, :, filt].max(axis=(1, 2))
#     th = np.percentile(max_vals, 90)
#     select_examples = max_vals >= th
#     input_data = [x[select_examples] for x in visualizer.test_data]
#     locs = np.argwhere(activations[select_examples, :, :, filt] ==
#                        max_vals[select_examples, None, None]).astype(np.int32)
#     fig, ax = plt.subplots(3, 5, sharex=True, constrained_layout=True, figsize=(18, 10))
#     fig.suptitle("filter {}, weight: {:5.3f}".format(filt, dense_weights[filt]))
#     for i, j in enumerate(np.random.choice(np.arange(locs.shape[0]), 15, False)):
#         grads = visualizer.get_gradients(relu_layers[select_layer].output[:, locs[j, 1], locs[j, 2], filt],
#                                          visualizer.model.inputs[0], [x[j, None] for x in input_data])
#         fov = np.any(grads[0] != 0, axis=2)
#         rows = np.any(fov, axis=1)
#         cols = np.any(fov, axis=0)
#         rmin, rmax = np.where(rows)[0][[0, -1]]
#         cmin, cmax = np.where(cols)[0][[0, -1]]
#         data = input_data[0][j, rmin:rmax, cmin:cmax]
#         ax[i // 5, i % 5].plot(data[:, :, 0].T, 'r')
#         ax[i // 5, i % 5].plot(data[:, :, 1].T, 'b')
#         ax[i // 5, i % 5].plot(data[:, :, 2].T, 'g')
#     plt.savefig("C:\\Users\\Greg\\Desktop\\semeter_meeting\\{}.png".format(filt))
#     plt.close(fig)
#     plt.clf()
