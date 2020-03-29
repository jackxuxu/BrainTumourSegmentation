import numpy as np
import tensorflow as tf
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


def create_data(in_path, out_path, verbose=True, min_max_norm=False, swapaxes=False):
    '''
    Function to read medical image from BRATS2015 and convert them into .npy
    :param in_path: input path where BRATS2015 is stored
    :param out_path: path where preprocessed data is subjected to stored
    :param verbose: output data tree example while processing the files
    :param min_max_norm: toggle for min max normalization
    :param swapaxes: swapaxes after stacking up pre-processed images, (slices, img_size, img_size) => (img_size, img_size, slices)
    :return: None (check output folder)
    '''
    total_patients = []
    # directory for training and testing
    for d_00 in sorted(os.listdir(in_path)):
        # ignore zip files
        if not d_00.endswith('.zip'):
            print(d_00)
            merge_d00 = os.path.join(in_path + d_00)
        # skip the loop for .zip extension
        else:
            continue
        # create file directory [Training, Testing]
        save_path_01 = (out_path + d_00 + '/')
        if not os.path.exists(save_path_01):
            os.makedirs(save_path_01)

        # training or testing > hgg or lgg
        for d_01 in sorted(os.listdir(merge_d00)):
            print(' ->', d_01)
            merge_d01 = os.path.join(merge_d00 + '/' + d_01)
            patient_counts = 0
            # create file directory [HGG, LGG]
            save_path_02 = (save_path_01 + d_01 + '/')
            if not os.path.exists(save_path_02):
                os.makedirs(save_path_02)

            for steps_01, d_02 in enumerate(sorted(os.listdir(merge_d01))):
                break_01 = 0
                # debug
                # list only the first dir
                if steps_01 == 0:
                    break_01 = 1
                    print('  -->', d_02)  # patient name
                #
                patient_counts += 1
                merge_d02 = os.path.join(merge_d01 + '/' + d_02)

                multimodal_name_list = []
                for steps_02, d_03 in enumerate(sorted(os.listdir(merge_d02))):
                    # create file
                    multimodal_file_name = d_03.split('.')[-2]  # MR_Flair, T2,..
                    multimodal_name_list.append(multimodal_file_name)
                    save_path_03 = (save_path_02 + multimodal_file_name + '/')
                    if not os.path.exists(save_path_03):
                        os.makedirs(save_path_03)
                    # debug
                    # list only the first dir
                    if break_01 == 1 and steps_02 != 5:
                        print('   --->', d_03)  # multimodal
                    #
                    merge_d03 = os.path.join(merge_d02 + '/' + d_03)
                    # read files with wild card .mha ending
                    med_img = glob.glob('{}/*.mha'.format(merge_d03))  # return list!
                    save_path_04 = (save_path_02 + multimodal_file_name + '/')
                    for mha in med_img:
                        read_med_img, _ = medpy.io.load(mha)

                    # min max normalization switch, label 'OT' not included
                    if multimodal_file_name != 'OT' and min_max_norm == True:
                        norm_list = []
                        for i in range(read_med_img.shape[-1]):  # last channel is the slices
                            max_val = np.max(read_med_img[:, :, i])
                            min_val = np.min(read_med_img[:, :, i])
                            norm = (read_med_img[:, :, i] - min_val) / (max_val - min_val)
                            norm_list.append(norm)

                        read_med_img = np.array(norm_list)  # shape(155,240,240)
                        read_med_img = np.nan_to_num(read_med_img)  # at times, max = 0, min = 0
                        if swapaxes == True:  # =>(240, 240, 155)
                            read_med_img = np.swapaxes(read_med_img, 0, 1)
                            read_med_img = np.swapaxes(read_med_img, 1, 2)
                    # file name => e.g. MR_Flair_brats_2013_pat0103_1.npy (multimodal + patient name)
                    np.save(save_path_04 + '{}_{}.npy'.format(multimodal_file_name, d_02), read_med_img)
                    #                 plt.imshow(read_med_img[:,:,20], cmap = 'gray')
            #                 plt.show()
            if verbose == True:
                print('*Number of patients: {}'.format(patient_counts))
                total_patients.append(patient_counts)
                print()

    if verbose == True:
        print()
        n_slices = 155
        t_patients = np.sum(total_patients)
        print('[Summary]')
        print('Total number of patients: {}'.format(t_patients))
        print('Total number of 2D images: {}'.format(t_patients * n_slices))
        print('  |_ Training: {}'.format((t_patients - total_patients[0]) * n_slices))
        print('  |_ Testing: {}'.format(total_patients[0] * n_slices))

def split_labels(inp_path, out_path):
    '''
    Function to split labels into 3 tumor segmentations => core, enhancing tumor, complete tumor

    :param inp_path: input path where the original labels in .npy is stored
    :param out_path: output path where the split labels should be stored
    :return: None (check output folder)
    '''
    for lbls in sorted(os.listdir(inp_path)):
        patient_tag = lbls.split('.')[0]
        npy_path = os.path.join(inp_path+'/'+lbls)
        labels = np.load(npy_path)
        #labels path
        core_path = out_path+'core/'
        enhancing_tumor_path = out_path+'enhancing_tumor/'
        complete_tumor_path = out_path+'complete_tumor/'

        #core
        if not os.path.exists(core_path):
            os.makedirs(core_path)
        core = np.where((labels==1) | (labels==3), labels, 0)
        np.save(core_path+'core_{}.npy'.format(patient_tag), core)
        #enhancing tumor
        if not os.path.exists(enhancing_tumor_path):
            os.makedirs(enhancing_tumor_path)
        enhancing_tumor = np.where(labels==4, labels, 0)
        np.save(enhancing_tumor_path+'enhancing_tumor_{}.npy'.format(patient_tag), enhancing_tumor)

        #complete tumor
        if not os.path.exists(complete_tumor_path):
            os.makedirs(complete_tumor_path)
        complete_tumor = np.where(labels<=1, labels, 1 )
        np.save(complete_tumor_path+'complete_tumor_{}.npy'.format(patient_tag), complete_tumor)


def threeD_to_twoD(save_dir, ds_list, ds_list_str):
    '''
    Stack all the images according to slices of a patient to create a 2D image stacks

    @param save_dir: The directory where the images need to be save
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for ds, file_name in zip(ds_list, ds_list_str):
        for data in sorted(os.listdir(ds)):
            if data == 'OT':
                continue  # skip the label directory
            merge_path_01 = os.path.join(ds + data + '/')
            img_stack = []
            for imgs in sorted(os.listdir(merge_path_01)):
                merge_path_02 = os.path.join(merge_path_01 + '/' + imgs)
                med_img = np.load(merge_path_02).astype(np.float32)
                for i in range(med_img.shape[-1]):  # last dimension is the number of slices
                    img_stack.append(med_img[:, :, i])
            img_stack = np.array(img_stack)
            np.save(save_dir + '{}_{}'.format(file_name, data), img_stack)


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


def serialize_example(image, label):
    '''
    Adding image and label info to TFRecords dataset
    '''
    feature = {
        'image': _bytes_feature(image),
        'label': _bytes_feature(label),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecords(tfrecord_dir, image_paths, labels_path):
    '''
    write TFRecords to appointed directory
    '''
    with tf.io.TFRecordWriter(tfrecord_dir) as writer:
        for image, label in zip(image_paths, labels_path):
            img_bytes = tf.io.serialize_tensor(image)
            labels = tf.io.serialize_tensor(label)
            example = serialize_example(img_bytes, labels)
            writer.write(example)


def read_tfrecord(serialized_example):
    '''
    read TFRecords from appointed directory
    '''
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=float)
    label = tf.io.parse_tensor(example['label'], out_type=float)

    return image, label


def parse_tfrecord(tf_dir):
    tfrecord_dataset = tf.data.TFRecordDataset(tf_dir)
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)
    return parsed_dataset