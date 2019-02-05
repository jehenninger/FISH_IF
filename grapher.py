import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt

# @TODO Add number of FISH spots. Change colors. Update graph title. Add x and y labels. Add colorbar. Add subplot title name
def make_2D_contour_plot(fish, protein, random, data, input_params):

    fig, ax = plt.subplots(1, 3)

    x = np.linspace(0, fish.shape[0], fish.shape[0])
    y = np.linspace(0, fish.shape[1], fish.shape[1])

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    ax[0].contourf(fish)

    ax[1].contourf(protein)

    ax[2].contourf(random)

    for a in ax:
        a.set_aspect('equal', 'box')

        # n = fish.shape[0]/6  # 6 corresponds to the number of axes tick divisions
        n = fish.shape[0]
        # tick_locations = [n/6, 2*n/6, 3*n/6, 4*n/6, 5*n/6]
        tick_locations = np.arange(0.0, n, 0.5/input_params.xy_um_per_px)

        tick_locations = tick_locations[1:]  # get rid of end labels

        l = input_params.b  # l corresponds to the um box edge length

        # I add the 0.01 here to make '0' not negative.
        tick_labels = ['{0:g}'.format(np.round((t*input_params.xy_um_per_px - l/2) + 0.01,1).item()) for t in tick_locations]

        a.set_xticks(tick_locations)
        a.set_xticklabels(tick_labels, {'fontsize': 8})

        a.set_yticks(tick_locations)
        a.set_yticklabels(tick_labels, {'fontsize': 8})

    # ax[0].contourf(xv, yv, fish)
    #
    # ax[1].contourf(xv, yv, protein)
    #
    # ax[2].contourf(xv, yv, random)

    plt.suptitle(data.sample_name)

    fig.tight_layout()

    plt.savefig(os.path.join(data.output_directories['summary'], data.sample_name + '_2D_contour.png'), dpi=300)

    plt.close()

