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
