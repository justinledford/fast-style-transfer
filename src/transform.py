from keras.layers import Conv2D, Conv2DTranspose, BatchNormalization
from keras import backend as K

# TODO: instance norm, init weights?
def net(image, tf_session):
    K.set_session(tf_session)

    conv1   = Conv2DBatchNorm(image, 32, 9)
    conv2   = Conv2DBatchNorm(conv1, 64, 3, strides=2)
    conv3   = Conv2DBatchNorm(conv2, 128, 3, strides=2)
    resid1  = Conv2DResidualBlock(conv3)
    resid2  = Conv2DResidualBlock(resid1)
    resid3  = Conv2DResidualBlock(resid2)
    resid4  = Conv2DResidualBlock(resid3)
    resid5  = Conv2DResidualBlock(resid4)
    conv_t1 = Conv2DTransposeBatchNorm(conv3, 64, 3, strides=2)
    conv_t2 = Conv2DTransposeBatchNorm(conv_t1, 32, 3, strides=2)
    conv_t3 = Conv2DBatchNorm(conv_t2, 3, 9, activation='tanh')
    preds = conv_t3 * 150 + 255./2
    return preds

def Conv2DBatchNorm(inputs, filters, kernel_size,
                    strides=1, activation='relu'):
    return BatchNormalization()(
            Conv2D(
                filters,
                (kernel_size, kernel_size),
                strides=strides,
                activation=activation,
                padding='same'
            )(inputs))

def Conv2DTransposeBatchNorm(inputs, filters, kernel_size,
                              strides=1, activation=None):
    return BatchNormalization()(
            Conv2DTranspose(
                filters,
                (kernel_size, kernel_size),
                strides=strides,
                activation=activation,
                padding='same'
            )(inputs))

def Conv2DResidualBlock(inputs):
    tmp     = Conv2DBatchNorm(inputs, 128, 3)
    tmp2    = Conv2DBatchNorm(tmp, 128, 3, activation=None)
    return keras.layers.add([tmp, tmp2])
