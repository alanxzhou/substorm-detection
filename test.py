import numpy as np
import pandas as pd
import anneal
from pymap3d.vincenty import vdist

region_corners = [[-130, 45], [-60, 70]]

all_stations = pd.read_csv("../data/supermag_stations.csv", index_col=0, usecols=[0, 1, 2, 5])
statloc = all_stations.values[:, :2]
statloc[statloc > 180] -= 360
region_mask = ((statloc[:, 0] > region_corners[0][0]) * (statloc[:, 0] < region_corners[1][0]) *
               (statloc[:, 1] > region_corners[0][1]) * (statloc[:, 1] < region_corners[1][1]))
stations = list(all_stations[region_mask].index)
statloc = statloc[region_mask]

a = statloc[:, None, :] * np.ones((1, statloc.shape[0], statloc.shape[1]))
locs = np.reshape(np.concatenate((a, np.transpose(a, [1, 0, 2])), axis=2), (-1, 4)).astype(float)

d, a1, a2 = vdist(locs[:, 1], locs[:, 0], locs[:, 3], locs[:, 2])
dists = d.reshape((statloc.shape[0], statloc.shape[0]))
dists[np.isnan(dists)] = 0

sa = anneal.SimAnneal(statloc, dists, stopping_iter=10000)
sa.anneal()
sa.visualize_routes()
