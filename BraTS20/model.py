import tensorflow as tf
from utils_model import *
from tensorflow.keras.layers import Conv2D

hn = 'he_normal' #kernel initializer


def Unet_model(input_layer, dropout=0.2):
    # downsampling
    #     conv1 = coordconv_block(input_layer, x_dim=240, y_dim=240, filters=64)
    conv1 = conv_block(input_layer, filters=64, kernel_initializer=hn)
    pool1 = pool(conv1)

    conv2 = conv_block(pool1, filters=128, kernel_initializer=hn)
    pool2 = pool(conv2)

    conv3 = conv_block(pool2, filters=256, kernel_initializer=hn)
    pool3 = pool(conv3)

    conv4 = conv_block(pool3, filters=512, kernel_initializer=hn, dropout_rate=dropout)
    pool4 = pool(conv4)

    conv5 = conv_block(pool4, filters=1024, kernel_initializer=hn, dropout_rate=dropout)

    # upsampling
    up1 = up(conv5, filters=512, merge=conv4, kernel_initializer=hn)
    #     conv6 = coordconv_block(up1, x_dim=30, y_dim=30, filters=512)
    conv6 = conv_block(up1, filters=512, kernel_initializer=hn)

    up2 = up(conv6, filters=256, merge=conv3, kernel_initializer=hn)
    conv7 = conv_block(up2, filters=256, kernel_initializer=hn)

    up3 = up(conv7, filters=128, merge=conv2, kernel_initializer=hn)
    conv8 = conv_block(up3, filters=128, kernel_initializer=hn)

    up4 = up(conv8, filters=64, merge=conv1, kernel_initializer=hn)
    conv9 = conv_block(up4, filters=64, kernel_initializer=hn)

    output_layer = Conv2D(4, (1, 1), activation='softmax')(conv9)

    return output_layer


def AttUnet_model(input_layer, attention_mode='grid', dropout=0.2):
    # downsampling path
    conv1 = conv_block(input_layer, filters=64, kernel_initializer=hn)
    pool1 = pool(conv1)

    conv2 = conv_block(pool1, filters=128, kernel_initializer=hn)
    pool2 = pool(conv2)

    conv3 = conv_block(pool2, filters=256, kernel_initializer=hn)
    pool3 = pool(conv3)

    conv4 = conv_block(pool3, filters=512, kernel_initializer=hn, dropout_rate=dropout)
    pool4 = pool(conv4)

    conv5 = conv_block(pool4, filters=1024, kernel_initializer=hn, dropout_rate=dropout)

    # upsampling path
    att01 = attention_block(conv4, conv5, 512)
    up1 = up(conv5, filters=512, merge=att01, kernel_initializer=hn)
    conv6 = conv_block(up1, filters=512, kernel_initializer=hn)

    if attention_mode == 'grid':
        att02 = attention_block(conv3, conv6, 256)
    else:
        att02 = attention_block(conv3, conv4, 256)
    up2 = up(conv6, filters=256, merge=att02, kernel_initializer=hn)
    conv7 = conv_block(up2, filters=256, kernel_initializer=hn)

    if attention_mode == 'grid':
        att03 = attention_block(conv2, conv7, 128)
    else:
        att03 = attention_block(conv2, conv3, 128)
    up3 = up(conv7, filters=128, merge=att03, kernel_initializer=hn)
    conv8 = conv_block(up3, filters=128, kernel_initializer=hn)

    if attention_mode == 'grid':
        att04 = attention_block(conv1, conv8, 64)
    else:
        att04 = attention_block(conv1, conv2, 64)
    up4 = up(conv8, filters=64, merge=att04, kernel_initializer=hn)
    conv9 = conv_block(up4, filters=64, kernel_initializer=hn)

    output_layer = Conv2D(4, (1, 1), activation='softmax')(conv9)

    return output_layer
