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
