import methods
import matplotlib

import numpy as np
import pandas as pd
import os
import sys

from matplotlib import pyplot as plt
from matplotlib import colors
from skimage import color, exposure

from mpl_toolkits.mplot3d import Axes3D

def make_2D_contour_plot(fish, protein, random, sample_name, protein_name, data, input_params, num_of_fish_spots):
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

    fish_contour = ax[0].contourf(fish, 15, cmap=fish_cmap, vmin=fish_min, vmax=fish_max)
    ax[0].set_title('FISH (n = ' + str(num_of_fish_spots) + ')', {'fontsize': master_font_size})
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

    protein_contour = ax[1].contourf(protein, 15, cmap=protein_cmap, vmin=protein_min, vmax=protein_max)
    ax[1].set_title('channel_' + protein_name + '_IF', {'fontsize': master_font_size})
    ax[1].set_xlabel('µm', {'fontsize': master_font_size})
    ax[1].set_ylabel('µm', {'fontsize': master_font_size})
    protein_cbar = plt.colorbar(protein_contour, ax=ax[1], fraction=colormap_fraction, pad=colormap_pad, aspect=colorbar_aspect)
    protein_cbar.ax.tick_params(labelsize=master_font_size)

    random_contour = ax[2].contourf(random, 15, cmap=protein_cmap, vmin=protein_min, vmax=protein_max)
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


def make_3D_surface_plot(fish, protein, random, sample_name, protein_name, data, input_params, num_of_fish_spots):
    master_font_size = 8
    colormap_fraction = 0.05
    colormap_pad = 0.08
    colorbar_aspect = 18

    protein_hex = ['#000000', '#18091f', '#250d38', '#330c52', '#41066f', '#480381',
                   '#3f0c7d', '#351179', '#2a1575', '#1e1871', '#37386d', '#4c6765',
                   '#529859', '#48ca44', '#00ff00']

    protein_rgb = []
    for h in protein_hex:
        protein_rgb.append(methods.hex_to_rgb(h))

    protein_cmap = colors.ListedColormap(protein_rgb, name='protein', N=15)

    fish_rgb = [1, 0, 1]
    fish_cmap = np.zeros((15, 3))
    for i in range(len(fish_rgb)):
        fish_cmap[:, i] = np.linspace(0, fish_rgb[i], 15)

    fish_cmap = colors.ListedColormap(fish_cmap, name='fish', N=15)  # Defaults to magenta

    fig = plt.figure(figsize=(10, 7.5))
    ax = []
    ax.append(fig.add_subplot(1, 3, 1, projection='3d'))
    ax.append(fig.add_subplot(1, 3, 2, projection='3d'))
    ax.append(fig.add_subplot(1, 3, 3, projection='3d'))

    fish_min = np.min(fish)
    fish_max = np.max(fish)

    x = y = np.arange(0, fish.shape[0], 1.0)
    X, Y = np.meshgrid(x, y)

    fish_surface = ax[0].plot_surface(X, Y, fish, cmap=fish_cmap, vmin=fish_min, vmax=fish_max, linewidth=0)
    ax[0].set_title('FISH (n = ' + str(num_of_fish_spots) + ')', {'fontsize': master_font_size})

    # fish_cbar = plt.colorbar(fish_surface, ax=ax[0], fraction=colormap_fraction, pad=colormap_pad,
    #                          aspect=colorbar_aspect, shrink=0.5)
    # fish_cbar.ax.tick_params(labelsize=master_font_size)

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

    protein_surface = ax[1].plot_surface(X, Y, protein, cmap=protein_cmap, vmin=protein_min, vmax=protein_max, linewidth=0)
    ax[1].set_title('channel_' + protein_name + '_IF', {'fontsize': master_font_size})
    ax[1].set_zlim(protein_min, protein_max)
    # protein_cbar = plt.colorbar(protein_surface, ax=ax[1], fraction=colormap_fraction, pad=colormap_pad,
    #                            aspect=colorbar_aspect, shrink=0.5)
    # protein_cbar.ax.tick_params(labelsize=master_font_size)

    random_surface = ax[2].plot_surface(X, Y, random, cmap=protein_cmap, vmin=protein_min, vmax=protein_max, linewidth=0)
    ax[2].set_title('random_IF', {'fontsize': master_font_size})
    ax[2].set_zlim(protein_min, protein_max)
    # random_cbar = plt.colorbar(protein_surface, ax=ax[2], fraction=colormap_fraction, pad=colormap_pad,
    #                           aspect=colorbar_aspect, shrink=0.5)
    # random_cbar.ax.tick_params(labelsize=master_font_size)

    for a in ax:
        a.set_aspect(0.5)
        a.view_init(26, 298)

        # make the panes transparent
        a.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        a.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        a.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # make the gridlines transparent
        a.xaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        a.yaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)
        a.zaxis._axinfo["grid"]['color'] = (1.0, 1.0, 1.0, 0.0)

        # n = fish.shape[0]

        # tick_locations = np.arange(0.0, n, 0.5 / input_params.xy_um_per_px)

        # tick_locations = tick_locations[1:]  # get rid of end labels

        # l = input_params.b  # l corresponds to the um box edge length

        # I add the 0.01 here to make '0' not negative.
        # tick_labels = ['{0:g}'.format(np.round((t * input_params.xy_um_per_px - l / 2) + 0.01, 1).item()) for t in
        #                tick_locations]

        # a.set_xticks(tick_locations)
        a.set_xticks([])
        a.set_yticks([])

        # a.set_xticklabels(tick_labels, {'fontsize': master_font_size})

        # a.set_yticks(tick_locations)
        # a.set_yticklabels(tick_labels, {'fontsize': master_font_size})

        for t in a.zaxis.get_major_ticks(): t.label.set_fontsize(8)

    plt.suptitle(sample_name + '_channel_' + protein_name)

    fig.tight_layout()

    plt.savefig(
        os.path.join(data.output_directories['summary'], sample_name + '_channel_' + protein_name + '_3D_surface.png'),
        dpi=300)
    plt.savefig(
        os.path.join(data.output_directories['summary'], sample_name + '_channel_' + protein_name + '_3D_surface.pdf'),
        transparent=True)

    plt.close()

def make_image_output(data, input_params):
    master_alpha = 0.5
    master_bg_label = 0
    master_text_offset = 5
    master_font_dict = {'fontsize': 8, 'color': 'w'}


    fig, ax = plt.subplots(2, 4, figsize=(10, 7.5))

    #
    # axis 0 = nucleus with label mask
    #
    nucleus_image = exposure.equalize_adapthist(methods.max_project(data.nucleus_image))

    nucleus_label_overlay = color.label2rgb(data.nuclear_binary_labeled, image=nucleus_image,
                                            alpha=master_alpha, image_alpha=1, bg_label=master_bg_label)

    ax[0, 0].imshow(nucleus_label_overlay)
    ax[0, 0].set_title('nucleus labels', {'fontsize': 8})

    ax[1, 0].imshow(nucleus_label_overlay)
    ax[1, 0].set_title('nucleus labels', {'fontsize': 8})

    #
    # axis 5 = nucleus with random points mask
    #

    nucleus_rand_label = np.full(shape=data.nuclear_binary_labeled.shape, fill_value=-1, dtype=int)

    for i, rand_spot in enumerate(data.rand_spots):
        nucleus_rand_label[rand_spot[1], rand_spot[2]] = i

    nucleus_rand_label_overlay = color.label2rgb(nucleus_rand_label, image=nucleus_image,
                                                 alpha=master_alpha, image_alpha=1, bg_label=-1)

    ax[1, 1].imshow(nucleus_rand_label_overlay)

    for i, rand_spot in enumerate(data.rand_spots):
        _, rand_center_r, rand_center_c = methods.get_spot_center(rand_spot)

        ax[1, 1].text(rand_center_c + master_text_offset, rand_center_r - master_text_offset, str(i), master_font_dict)

    ax[1, 1].set_title('random points', {'fontsize':8})

    #
    # axis 1 = FISH with called FISH spot mask
    #

    fish_label = np.full(shape=data.nuclear_binary_labeled.shape, fill_value=-1, dtype=int)

    for i, fish_spot in enumerate(data.fish_spots):
        fish_label[fish_spot[1], fish_spot[2]] = i

    fish_image = exposure.equalize_adapthist(methods.max_project(data.fish_image))

    fish_label_overlay = color.label2rgb(fish_label, image=fish_image,
                                         alpha=master_alpha, image_alpha=1, bg_label=-1)

    ax[0, 1].imshow(fish_label_overlay)

    for i, fish_center in enumerate(data.fish_centers):
        ax[0, 1].text(fish_center[2] + master_text_offset, fish_center[1] - master_text_offset, str(i), master_font_dict)

    ax[0, 1].set_title('FISH spots', {'fontsize':8})

    #
    # axis 2 (and 3) = IF with FISH mask. Two channels if there are multiple IFs
    #

    protein_image = data.protein_images[0]
    protein_image = exposure.equalize_adapthist(methods.max_project(protein_image))

    IF_fish_label_overlay = color.label2rgb(fish_label, image=protein_image,
                                            alpha=master_alpha, image_alpha=1, bg_label=-1)
    ax[0, 2].imshow(IF_fish_label_overlay)
    ax[0, 2].set_title('IF_channel_' + str(data.protein_channel_names[0]), {'fontsize': 8})

    if input_params.multiple_IF_flag:
        protein_image_b = data.protein_images[1]
        protein_image_b = exposure.equalize_adapthist(methods.max_project(protein_image_b))

        IF_fish_label_overlay_b = color.label2rgb(fish_label, image=protein_image_b,
                                                  alpha=master_alpha, image_alpha=1, bg_label=-1)
        ax[0, 3].imshow(IF_fish_label_overlay_b)
        ax[0, 3].set_title('IF_channel_' + str(data.protein_channel_names[1]), {'fontsize':8})

    #
    # axis 6 (and 7) = IF with random points. Two channels if there are multiple IFs
    #
    IF_rand_label_overlay = color.label2rgb(nucleus_rand_label, image=protein_image,
                                            alpha=master_alpha, image_alpha=1, bg_label=-1)
    ax[1, 2].imshow(IF_rand_label_overlay)
    ax[1, 2].set_title('IF_channel_' + str(data.protein_channel_names[0]), {'fontsize': 8})

    if input_params.multiple_IF_flag:
        IF_rand_label_overlay_b = color.label2rgb(nucleus_rand_label, image=protein_image_b,
                                                  alpha=master_alpha, image_alpha=1, bg_label=-1)
        ax[1, 3].imshow(IF_rand_label_overlay_b)
        ax[1, 3].set_title('IF_channel_' + str(data.protein_channel_names[1]), {'fontsize': 8})

    for i in range(2):
        for j in range(4):
            methods.clear_axis_ticks(ax[i, j])
            ax[i, j].set_aspect('equal', 'box')

    plt.suptitle(data.sample_name, fontsize=10)
    plt.tight_layout()

    plt.savefig(
        os.path.join(data.output_directories['individual_images'], data.sample_name + '_segmentation.png'),
        dpi=300)
    # plt.savefig(
    #     os.path.join(data.output_directories['individual_images'], data.sample_name + '_segmentation.pdf'),
    #     transparent=True)

    plt.close()

