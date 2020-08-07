import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import medpy.io

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

def concat_recursive(a, b, max_count, count=0):
    '''
    Recursively concatenate the image stacks with the next image stacks

    @param a: Top first image stacks
    @param b: Following image stacks
    '''
    while count < max_count - 1:
        c = np.concatenate((a, b), axis=0)
        a = c
        count += 1
        concat_recursive(a, b, max_count, count)
    return a


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image):
    '''
    Adding image and label info to TFRecords dataset
    '''
    feature = {
        'image': _bytes_feature(image)
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(tfrecord_dir, image_paths):
    '''
    write TFRecords to appointed directory
    '''
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for image in image_paths:
            img_bytes = tf.io.serialize_tensor(image)
            example = serialize_example(img_bytes)
            writer.write(example)


def read_tfrecord(serialized_example):
    '''
    read TFRecords from appointed directory
    '''
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string)
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.parse_tensor(example['image'], out_type=float)
    return image


def parse_tfrecord(tf_dir):
    tfrecord_dataset = tf.data.TFRecordDataset(tf_dir)
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    return parsed_dataset


def min_max_norm(slice):
    "Min max norm channel wise"
    max_channel = np.max(slice)
    min_channel = np.min(slice)
    norm = (slice-min_channel)/(max_channel - min_channel)
    return norm

def std_norm(slice):
    """
    Removes 1% of the top and bottom intensities and perform
    normalization on the input 2D slice.
    """
    b = np.percentile(slice, 99)
    t = np.percentile(slice, 1)
    slice = np.clip(slice, t, b)
    if np.std(slice)==0:
        return slice
    else:
        slice = (slice - np.mean(slice)) / np.std(slice)
        return slice

def normalize_modalities(Slice, mode = None):
    """
    Performs normalization on each modalities of input
    """
    assert mode!=None, "Please in put [mode] type! 'std' for standard normalization, 'minmax' for minmax normalization"
    normalized_slices = np.zeros_like(Slice).astype(np.float32)
    for slice_ix in range(4):
        if mode=='std':
            normalized_slices[..., slice_ix] = std_norm(Slice[..., slice_ix])
        if mode=='minmax':
            normalized_slices[..., slice_ix] = min_max_norm(Slice[..., slice_ix])
    return normalized_slices

def dicesq(y_true, y_pred, smooth=1e-5):
    '''
    Modified dice coefficient as refer to: https://arxiv.org/abs/1606.04797
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :return: Modified dice coefficient
    '''
    nmr = 2*tf.reduce_sum(y_true*y_pred)
    dnmr = tf.reduce_sum(y_true**2) + tf.reduce_sum(y_pred**2) + smooth
    return (nmr / dnmr)

def dicesq_loss(y_true, y_pred):
    '''
    Modified dice coefficient loss
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    '''
    return 1- dicesq(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1e-5):
    '''
    Dice coefficient for tensorflow
    :param y_true: Ground truth
    :param y_pred: Prediction from the model
    :return: dice coefficient
    '''
    #if input is not flatten
    if (tf.rank(y_true)!=1 and tf.rank(y_pred)!=1):
        y_true = tf.reshape(y_true, [-1]) #flatten
        y_pred = tf.reshape(y_pred, [-1]) #flatten
    #casting for label from int32 to float32 for computation
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / \
(tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_coef_loss(y_true, y_pred):
    '''
    Dice coefficient loss for IOU
    '''
    return 1-dice_coef(y_true, y_pred)