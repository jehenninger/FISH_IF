import numpy as np
import pandas as pd
import os
import sys
from matplotlib import pyplot as plt

# @TODO Change colors. Add colorbar
def make_2D_contour_plot(fish, protein, random, sample_name, protein_name, data, input_params):
    
    master_font_size = 8

    fig, ax = plt.subplots(1, 3)

    ax[0].contourf(fish)
    ax[0].set_title('FISH (n = ' + str(len(data.fish_spots)), {'fontsize': master_font_size})
    ax[0].set_xlabel('µM', {'fontsize': master_font_size})
    ax[0].set_ylabel('µM', {'fontsize': master_font_size})

    ax[1].contourf(protein)
    ax[1].set_title('channel_' + protein_name + '_IF', {'fontsize': master_font_size})
    ax[1].set_xlabel('µM', {'fontsize': master_font_size})
    ax[1].set_ylabel('µM', {'fontsize': master_font_size})

    ax[2].contourf(random)
    ax[2].set_title('random_IF', {'fontsize': master_font_size})
    ax[2].set_xlabel('µM', {'fontsize': master_font_size})
    ax[2].set_ylabel('µM', {'fontsize': master_font_size})

    for a in ax:
        a.set_aspect('equal', 'box')

        n = fish.shape[0]

        tick_locations = np.arange(0.0, n, 0.5/input_params.xy_um_per_px)

        tick_locations = tick_locations[1:]  # get rid of end labels

        l = input_params.b  # l corresponds to the um box edge length

        # I add the 0.01 here to make '0' not negative.
        tick_labels = ['{0:g}'.format(np.round((t*input_params.xy_um_per_px - l/2) + 0.01,1).item()) for t in tick_locations]

        a.set_xticks(tick_locations)
        a.set_xticklabels(tick_labels, {'fontsize': master_font_size})

        a.set_yticks(tick_locations)
        a.set_yticklabels(tick_labels, {'fontsize': master_font_size})

    plt.suptitle(sample_name + '_channel_' + protein_name)

    fig.tight_layout()

    plt.savefig(os.path.join(data.output_directories['summary'], sample_name + '_channel_' + protein_name + '_2D_contour.png'), dpi=300)
    plt.savefig(os.path.join(data.output_directories['summary'], sample_name + '_channel_' + protein_name + '_2D_contour.eps'))

    plt.close()

