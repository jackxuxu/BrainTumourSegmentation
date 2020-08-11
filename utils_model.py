import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, MaxPooling2D, UpSampling2D, Dropout
from tensorflow.keras.layers import InputLayer, Conv2DTranspose, Activation, BatchNormalization, Input
from coord_conv import CoordConv
from tensorflow.keras import Model


def conv_block(x_in, filters, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x


def coordconv_block(x_in, x_dim, y_dim, filters, batch_norm=False, kernel_size=(3, 3),
                    kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None, with_r=False):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    x = CoordConv(x_dim, y_dim, with_r, filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x


def conv_2d(x_in, filters, batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x


def pool(x_in, pool_size=(2, 2), type='Max'):
    if type == 'Max':
        p = MaxPooling2D(pool_size)(x_in)
    return p


def up(x_in, filters, merge, batch_norm=False,
       kernel_initializer='glorot_uniform', dropout_rate=None, size=(2, 2)):
    u = UpSampling2D(size)(x_in)
    conv = conv_2d(u, filters, batch_norm, acti='relu', kernel_initializer=kernel_initializer,
                   dropout_rate=dropout_rate)
    concat = tf.concat([merge, conv], axis=-1)
    return concat

hn = 'he_normal' #kernel initializer

def Unet_model(input_layer, dropout):
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
