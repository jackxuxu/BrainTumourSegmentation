import tensorflow as tf
from utils_model import *
from tensorflow.keras.layers import Conv2D, UpSampling2D, Activation, Add, Multiply
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Softmax


def attention_block(input_signal, gated_signal, filters, att_layer_name):
    # input signal feature maps
    is_fm = Conv2D(filters, kernel_size=(1,1), strides=(2, 2), padding = 'same')(input_signal)
    # gated signal feature maps
    gs_fm = Conv2D(filters, kernel_size=(1,1), strides=(1, 1), padding = 'same')(gated_signal)
    # debugger
    assert is_fm.shape!=gs_fm.shape, "Feature maps shape doesn't match!"
    # element wise sum
    add = Add()([is_fm, gs_fm])
    acti = Activation('relu')(add)
    # downsampled attention coefficient
    bottle_neck = Conv2D(1, kernel_size=(1,1), activation='sigmoid', name=att_layer_name)(acti)
    # bilinear interpolation to get attention coeffcient
    alpha = UpSampling2D(interpolation='bilinear')(bottle_neck)
    # filter off input signal's features with attention coefficient
    multi = Multiply()([input_signal, alpha])
    return multi, alpha


def PAM(inp_feature, layer_name, kernel_initializer='glorot_uniform', acti='relu'):
    '''
    Position attention module
    by default input shape => [w,h,c],[240, 240, 128] hence c/8 = 16
    :param layer_name: List of layer names
    [1st conv block, 2nd conv block, softmax output, 3rd conv block, position coefficient, Add output]
    :param inp_feature: feature maps of res block after up sampling [w,h,c]
    :return: PAM features [w/4,h/4,c]
    '''
    # downsampling by scale of 4 for GPU memory issue
    inp_feature = Conv2D(filters=128,kernel_size=1, strides=2, padding='same',activation='relu')(inp_feature)
    inp_feature = Conv2D(filters=128,kernel_size=1, strides=2, padding='same',activation='relu')(inp_feature)
    # dimensions
    b,w,h,c = inp_feature.shape
    # scale down ratio
    c_8 = c//8
    #
    assert len(layer_name)>=5, 'Layer list length should be 5!'
    # Branch01 Dimension: [w,h,c/8] => [(wxh),c/8]
    query = conv_2d(inp_feature, filters=c_8, layer_name=layer_name[0], batch_norm=False, kernel_size=(1, 1), acti=acti,
            kernel_initializer=kernel_initializer, dropout_rate=None)
    query = tf.reshape(query,[-1,(w*h),c_8 ])
    # Branch02 Dimension: [w,h,c/8] => [c/8,(wxh)]
    key = conv_2d(inp_feature, filters=c_8, layer_name=layer_name[1], batch_norm=False, kernel_size=(1, 1), acti=acti,
        kernel_initializer=kernel_initializer, dropout_rate=None)
    key = tf.reshape(key, [-1,(w*h),c_8 ])
    key = tf.einsum('bij->bji', key) # transpose/permutation
    # matmul pipeline 01 & 02
    matmul_0102 = tf.einsum('bij,bjk->bik', query, key) # [(wxh),(wxh)]
    #attention coefficient
    alpha_p = Softmax(name=layer_name[2])(matmul_0102) # [(wxh),(wxh)]
    # Branch03
    value = conv_2d(inp_feature, filters=c, layer_name=layer_name[3], batch_norm=False, kernel_size=(1, 1), acti=acti,
        kernel_initializer=kernel_initializer, dropout_rate=None)
    value = tf.reshape(value,[-1,(w*h),c]) # [(wxh),c]
    matmul_all = tf.einsum('bij,bjk->bik',alpha_p,value) # [(wxh),c]
    # Output
    output = tf.reshape(matmul_all, [-1,w,h,c]) # [w,h,c]
    # learnable coefficient to control the importance of CAM
    lambda_p = Conv2D(filters=1,kernel_size=1, padding='same',activation='sigmoid', name=layer_name[4])(inp_feature)
    output = Multiply()([output, lambda_p])
    output_add = Add(name = layer_name[-1])([output, inp_feature])
    return output_add


def CAM(inp_feature, layer_name):
    '''
    Channel attention module
    by default input shape => [w,h,c],[240, 240, 128] hence c/8 = 16
    :param inp_feature: feature maps of res block after up sampling [w,h,c]k
    :param layer_name: List of layer names
        [softmax output, channel attention coefficients, Add output]
    :return: CAM features [w/4,h/4,c]
    '''
    # downsampling by scale of 4 for GPU memory issue
    inp_feature = Conv2D(filters=128,kernel_size=1, strides=2, padding='same',activation='relu')(inp_feature)
    inp_feature = Conv2D(filters=128,kernel_size=1, strides=2, padding='same',activation='relu')(inp_feature)
    # dimensions
    b,w,h,c = inp_feature.shape
    # learnable coefficient to control the importance of CAM
    assert len(layer_name)>=2, 'Layer list length should be 2!'
    # Branch01 Dimension: [w,h,c] => [(wxh),c]
    query = tf.reshape(inp_feature, [-1,(w*h),c])
    # Branch02 Dimension: [w,h,c] => [c,(wxh)]
    key = tf.reshape(inp_feature, [-1,(w*h),c]) # [(wxh),c]
    key = tf.einsum('ijk->ikj', key) # Permute:[c,(wxh)]
    # matmul pipeline 01 & 02
    matmul_0201 = tf.einsum('ijk,ikl->ijl', key, query) # [c,c]
    #attention coefficient
    alpha_c = Softmax(name=layer_name[0])(matmul_0201) # [c,c]
    # Branch03 Dimension: [w,h,c] => [c,(wxh)]
    value = tf.reshape(inp_feature,[-1,(w*h),c]) # [(wxh),c]
    matmul_all = tf.einsum('ijk,ikl->ijl', value, alpha_c) # [(wxh),c]
    # output
    output = tf.reshape(matmul_all,[-1,w,h,c])# [w,h,c]
    # provides learnable parameter
    # *channel wise attention, inspired by Squeeze Excitation(SE) block
    GAP = GlobalAveragePooling2D()(output)
    dense01 = Dense(c//8, activation='relu')(GAP)
    lambda_c = Dense(c,activation='sigmoid', name=layer_name[1])(dense01)
    # outputs
    output = Multiply()([output, lambda_c])
    output_add = Add(name=layer_name[-1])([output, inp_feature])
    return output_add
