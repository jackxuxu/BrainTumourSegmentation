import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import medpy.io


def plot_comparison(input_img, caption, n_row=1, n_col=2, figsize=(5, 5), cmap='gray'):
    '''
    Plot comparison of multiple image but only in column wise!
    :param input_img: Input image list
    :param caption: Input caption list
    :param IMG_SIZE: Image size
    :param n_row: Number of row is 1 by DEFAULT
    :param n_col: Number of columns
    :param figsize: Figure size during plotting
    :return: Plot of (n_row, n_col)
    '''
    print()
    assert len(caption) == len(input_img), "Caption length and input image length does not match"
    assert len(input_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        axes[i].imshow(np.squeeze(input_img[i]), cmap=cmap)
        axes[i].set_xlabel(caption[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()


def plot_hist(inp_img, titles, n_row=1, n_col=2,
              n_bin=20, ranges=[0, 1], figsize=(5, 5)):
    '''
    Plot histogram side by side
    :param inp_img: Input image stacks as list
    :param titles: Input titles as list
    :param n_row: Number of row by DEFAULT 1
    :param n_col: Number of columns by DEFAULT 2
    :param n_bin: Number of bins by DEFAULT 20
    :param ranges: Range of pixel values by DEFAULT [0,1]
    :param figsize: Figure size while plotting by DEFAULT (5,5)
    :return:
        Plot of histograms
    '''
    assert len(titles) == len(inp_img), "Caption length and input image length does not match"
    assert len(inp_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        inp = np.squeeze(inp_img[i])
        axes[i].hist(inp.ravel(), n_bin, ranges)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def min_max_norm(images):
    """
    Min max normalization of images
    Parameters:
        images: Input stacked image list
    Return:
        Image list after min max normalization
    """
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi) / (m - mi)
    return images


def channel_standardization(image):
    '''
    Stanadrdization of image channel wise => Standard score
    Parameters:
        image: Input image
    Return:
        Standardized image, s.t. (pixel_value -)
    '''
    mean_val = np.mean(image, axis=-1)
    std_dev_val = np.std(image, axis=-1)
    output = (image - np.expand_dims(mean_val, axis=-1)) / (np.expand_dims(std_dev_val, axis=-1))
    # some val for std.dev = 0
    cast = np.nan_to_num(output)

    return cast