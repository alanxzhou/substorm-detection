import matplotlib.pyplot as plt
from detection.analysis.visualizations import Visualizer
from sklearn.cluster import AgglomerativeClustering
import keras.backend as K
import keras
import numpy as np
from vis import visualization
from vis import grad_modifiers
from vis import input_modifiers

TRAIN = False

data_fn = "../../data/2classes_data128.npz"
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
clust = AgglomerativeClustering(n_clusters=5, linkage="average", affinity="precomputed")
clust.fit(np.exp(-1 * cov))
clusters = np.argsort(clust.labels_)

# fig = plt.figure(figsize=(12, 12))
# plt.pcolormesh(cov[clusters][:, clusters], cmap='coolwarm')
# plt.xticks(np.arange(32), clusters)
# plt.yticks(np.arange(32), clusters)
# plt.savefig("C:\\Users\\Greg\\Desktop\\semeter_meeting\\correlation.png")
# plt.clf()
# plt.close(fig)

# for each filter:
#   optimize for highly activating input (feature visualization)
#   get examples (15 or so) of highly activating input data
#   for 2 or 3 different neurons, get examples of highly activating inputs, look to see what features are translated
filt = 15
tensor = -1 * visualizer.model.layers[27].output[0, 20, 5, filt]
for i in range(5):
    img = visualization.visualize_relationship(visualizer.model.input[0], tensor, max_iter=1000,
                                               seed_input=(np.random.rand(85, 96, 3)) * 5 - 80,
                                               grad_modifier=grad_modifiers.blur_size(3))

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=True, constrained_layout=True)
    for channel in range(3):
        ax[channel].pcolormesh(img[36:50, :, channel], cmap='coolwarm', vmin=img.min(), vmax=img.max())
        cb = ax[channel].pcolormesh(img[36:50, :, channel], cmap='coolwarm', vmin=img.min(), vmax=img.max())
        ax[channel].pcolormesh(img[36:50, :, channel], cmap='coolwarm', vmin=img.min(), vmax=img.max())
    fig.colorbar(cb, ax=ax.ravel().tolist())

max_vals = activations[:, 20, 5, filt]
th = np.percentile(max_vals, 95)
select_examples = max_vals >= th
input_data = [x[select_examples] for x in visualizer.test_data]
data = input_data[0]
for i in np.random.choice(np.arange(data.shape[0]), 5):
    fig, ax = plt.subplots(3, 1, sharex=True, constrained_layout=True)
    for channel in range(3):
        img = data[i, 36:50]
        ax[channel].pcolormesh(img[:, :, channel], cmap='coolwarm', vmin=img.min(), vmax=img.max())
        cb = ax[channel].pcolormesh(img[:, :, channel], cmap='coolwarm', vmin=img.min(), vmax=img.max())
        ax[channel].pcolormesh(img[:, :, channel], cmap='coolwarm', vmin=img.min(), vmax=img.max())
    fig.colorbar(cb, ax=ax.ravel().tolist())

plt.show()
