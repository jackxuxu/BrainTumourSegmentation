import tensorflow as tf
from coord_conv import CoordConv
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, Add, Multiply, MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, Dropout



hn = 'he_normal' #kernel initializer

def conv_block(x_in, filters, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    x = Conv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer)(x_in)
    # if batch_norm == True:
    #     x = BatchNormalization()(x)
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
    # if batch_norm == True:
    #     x = BatchNormalization()(x)
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

def attention_block(input_signal, gated_signal, filters):
    #input signal feature maps
    is_fm = Conv2D(filters, kernel_size=(1,1), strides=(2, 2), padding = 'same')(input_signal)
    #gated signal feature maps
    gs_fm = Conv2D(filters, kernel_size=(1,1), strides=(1, 1), padding = 'same')(gated_signal)
    #debugger
    assert is_fm.shape!=gs_fm.shape, "Feature maps shape doesn't match!"
    #element wise sum
    add = Add()([is_fm, gs_fm])
    acti = Activation('relu')(add)
    #downsampled attention coefficient
    bottle_neck = Conv2D(1, kernel_size=(1,1), activation='sigmoid')(acti)
    #bilinear interpolation to get attention coeffcient
    alpha = UpSampling2D(interpolation='bilinear')(bottle_neck)
    #filter off input signal's features with attention coefficient
    multi = Multiply()([input_signal, alpha])
    return multi

def conv_block_sep(x_in, filters, layer_name, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    assert type(filters)==list, "Please input filters of type list."
    assert type(layer_name)==list, "Please input filters of type list."
    x = SeparableConv2D(filters[0], kernel_size, padding='same', kernel_initializer=kernel_initializer, name = layer_name[0])(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    x = SeparableConv2D(filters[1], kernel_size, padding='same', kernel_initializer=kernel_initializer, name = layer_name[1])(x)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def conv_2d_sep(x_in, filters, layer_name, batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None):
    x = SeparableConv2D(filters, kernel_size, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def conv_2d(x_in, filters, layer_name, strides=(1,1), batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None):
    x = Conv2D(filters, kernel_size, strides, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def down_sampling_sep(x_in, filters, layer_name, batch_norm=False, kernel_size=(3, 3), acti='relu',
            kernel_initializer='glorot_uniform', dropout_rate=None, mode ='coord', x_dim=None, y_dim=None):
    assert mode=='coord' or mode=='normal', "Use 'coord' or 'normal' for mode!"
    if mode=='coord':
        #seperable coordconv
        assert (x_dim!=None and y_dim!=None), "Please input dimension for CoordConv!"
        x = Conv2D(1, kernel_size, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer)(x_in)
        x = CoordConv(x_dim=x_dim, y_dim=y_dim, with_r=False, filters=filters, strides=(1,1),
                      kernel_size = 3, padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x)
    else:
        #normal mode
        x = SeparableConv2D(filters, kernel_size, strides=(2, 2), padding='same', kernel_initializer=kernel_initializer, name=layer_name)(x_in)
    if batch_norm == True:
        x = BatchNormalization()(x)
    x = Activation(acti)(x)
    if dropout_rate != None:
        x = Dropout(dropout_rate)(x)
    return x

def res_block_sep(x_in, filters,  layer_name, batch_norm=False, kernel_size=(3, 3),
               kernel_initializer='glorot_uniform', acti='relu', dropout_rate=None):
    assert len(filters)==2, "Please assure that there is 3 values for filters."
    assert len(layer_name)==3, "Please assure that there is 3 values for layer name"
    layer_name_conv = [layer_name[i] for i in range(len(layer_name)-1)]
    output_conv_block = conv_block_sep(x_in, filters, layer_name_conv, batch_norm=batch_norm, kernel_size=kernel_size,
                                   kernel_initializer = kernel_initializer, acti = acti, dropout_rate=dropout_rate)
    output_add = Add(name = layer_name[-1])([output_conv_block, x_in])
    return output_add


