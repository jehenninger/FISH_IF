import methods
import matplotlib

import numpy as np
import pandas as pd
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import colors

# @TODO Change colors. Add colorbar
def make_2D_contour_plot(fish, protein, random, sample_name, protein_name, data, input_params):
    master_font_size = 8
    colormap_fraction = 0.05
    colormap_pad = 0.08
    colorbar_aspect = 18


    protein_hex = ['#000000','#18091f','#250d38','#330c52','#41066f','#480381',
                   '#3f0c7d','#351179','#2a1575','#1e1871','#37386d','#4c6765',
                   '#529859','#48ca44','#00ff00']

    protein_rgb = []
    for h in protein_hex:
        protein_rgb.append(methods.hex_to_rgb(h))

    protein_cmap = colors.ListedColormap(protein_rgb, name='protein', N=15)

    fish_rgb = [1, 0, 1]
    fish_cmap = np.zeros((15, 3))
    for i in range(len(fish_rgb)):
        fish_cmap[:, i] = np.linspace(0, fish_rgb[i], 15)

    fish_cmap = colors.ListedColormap(fish_cmap, name='fish', N=15)  # Defaults to magenta

    fig, ax = plt.subplots(1, 3, figsize=(10,7.5))


    fish_min = np.min(fish)
    fish_max = np.max(fish)

    fish_contour = ax[0].contourf(fish, cmap=fish_cmap, vmin=fish_min, vmax=fish_max)
    ax[0].set_title('FISH (n = ' + str(len(data.fish_spots)) + ')', {'fontsize': master_font_size})
    ax[0].set_xlabel('µm', {'fontsize': master_font_size})
    ax[0].set_ylabel('µm', {'fontsize': master_font_size})

    fish_cbar = plt.colorbar(fish_contour, ax=ax[0], fraction=colormap_fraction, pad=colormap_pad, aspect=colorbar_aspect)
    fish_cbar.ax.tick_params(labelsize=master_font_size)

    # find min and max of combined protein and random to get good scale
    p_max = np.max(protein)
    p_min = np.min(protein)
    r_max = np.max(random)
    r_min = np.min(random)



    if p_max > r_max:
        protein_max = p_max
    else:
        protein_max = r_max

    if p_min < r_min:
        protein_min = p_min
    else:
        protein_min = r_min

    protein_contour = ax[1].contourf(protein, cmap=protein_cmap, vmin=protein_min, vmax=protein_max)
    ax[1].set_title('channel_' + protein_name + '_IF', {'fontsize': master_font_size})
    ax[1].set_xlabel('µm', {'fontsize': master_font_size})
    ax[1].set_ylabel('µm', {'fontsize': master_font_size})
    protein_cbar = plt.colorbar(protein_contour, ax=ax[1], fraction=colormap_fraction, pad=colormap_pad, aspect=colorbar_aspect)
    protein_cbar.ax.tick_params(labelsize=master_font_size)

    random_contour = ax[2].contourf(random, cmap=protein_cmap, vmin=protein_min, vmax=protein_max)
    ax[2].set_title('random_IF', {'fontsize': master_font_size})
    ax[2].set_xlabel('µm', {'fontsize': master_font_size})
    ax[2].set_ylabel('µm', {'fontsize': master_font_size})
    random_cbar = plt.colorbar(protein_contour, ax=ax[2], fraction=colormap_fraction, pad=colormap_pad, aspect=colorbar_aspect)
    random_cbar.ax.tick_params(labelsize=master_font_size)

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
    plt.savefig(os.path.join(data.output_directories['summary'], sample_name + '_channel_' + protein_name + '_2D_contour.pdf'), transparent=True)

    plt.close()

