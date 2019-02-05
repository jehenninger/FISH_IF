import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt


def make_2D_contour_plot(fish, protein, random, data, input_params):

    fig, ax = plt.subplots(1, 2)

    x = np.linspace(0, fish.shape[0], fish.shape[0])
    y = np.linspace(0, fish.shape[1], fish.shape[1])

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    ax[0].contourf([xv, yv], fish)

    ax[1].contourf([xv, yv], protein)

    ax[2].contourf([xv, yv], random)

    plt.suptitle(data.sample_name)

    plt.savefig(os.path.join(data.output_directories['summary'], data.sample_name + '_2D_contour.png'), dpi=300)

