import tensorflow as tf
print(tf.version)
from tensorflow import keras
import horovod.tensorflow as hvd
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import models, losses
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.optimizers import Optimizer
#----------------------------------------------------------------
# Build base U-net architecture
#----------------------------------------------------------------
# Non-linear Activation switch - 0: linear; 1: non-linear
# If non-linear, then Leaky ReLU Activation function with alpha value as input
def actv_swtch(swtch, alpha_val):
    if swtch == 0:
        actv = "linear"
    else:
        actv = layers.LeakyReLU(alpha=alpha_val)
    return actv

class ReflectPadding2D(Layer):
    def __init__(self, pad_size, **kwargs):
        self.pad_size = pad_size
        super(ReflectPadding2D, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.pad(inputs, [[0, 0], [self.pad_size, self.pad_size], [self.pad_size, self.pad_size], [0, 0]], mode='REFLECT')

    def get_config(self):
        config = super(ReflectPadding2D, self).get_config()
        config.update({"pad_size": self.pad_size})
        return config
# Defining a generic 3D convolution layer for our use
# Inputs: [input tensor, output feature maps, filter size, dilation rate, stride,
# activation switch, if actv switch is 1 then activation-LReLU-alpha value, 
# regularizer factor]
def con2d(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):
    # Manually apply reflect padding
    pad_size = fil // 2
    # Use ReflectPadding2D instead of custom padding directly
    inp_padded = ReflectPadding2D(pad_size)(inp)
    # Apply the convolutional layer with 'valid' padding since we've already padded the input
    return layers.Conv2D(n_out, (fil, fil), dilation_rate=(dil_rate, dil_rate),
                                  strides=std,
                                  activation=actv_swtch(swtch, alpha_val),
                                  padding="valid", #padding="same",
                                  use_bias=True,
                                  kernel_regularizer=l2(reg_val),
                                  bias_regularizer=l2(reg_val))(inp_padded)

# Residual convolution block
def Res_conv_block(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):

    y = con2d(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    # Residual connection
    y = layers.Add()([y, con2d(inp, n_out, 1, dil_rate, std, swtch, alpha_val, reg_val)])

    return y

# Convolution downsampling block
def Conv_down_block(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):

    # Downsampling using the stride 2
    y = con2d(inp, n_out, fil, dil_rate, 2, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    return y

# Attention block
def Attention(inp, num_heads, key_dim):

    layer = layers.MultiHeadAttention(num_heads, key_dim, attention_axes=None)
    y = layer(inp, inp)

    return y

# Convolution upsampling block using bilinear interpolation
def Conv_up_block(inp, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val):

    # Upsampling using the stride 2 with transpose convolution
    y = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(inp)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = con2d(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    return y

#2D Pooling layer - inputs[0:avg;1:max, input tensor, pool size]
def pool2d(i, inp, ps):
    if i == 0:
        return layers.AveragePooling2D(pool_size=(ps, ps),
                                            strides=None,
                                            padding='same',
                                            data_format=None)(inp)
    else:
        return layers.MaxPooling2D(pool_size=(ps, ps),
                                            strides=None,
                                            padding='same',
                                            data_format=None)(inp)


# Generator architecture
def build_generator(inp_lat, inp_lon, out_lat, out_lon, chnl, out_vars, fil, dil_rate, std, swtch, alpha_val, reg_val, num_heads, key_dim):

    inp = layers.Input(shape=(inp_lat, inp_lon, chnl))
    y_st = layers.Input(shape=(out_lat, out_lon, 2))

    y_noise = layers.Input(shape=(out_lat, out_lon, 64))

    # Interpolate the input to target shape using bilinear interpolation
    y = tf.image.resize(inp, [out_lat, out_lon], method='bilinear')

    y = layers.Concatenate(axis=-1)([y, y_st, y_noise])

    # Encoding path
    skips = []
    for n_out in [64, 128, 256]:
        y = Res_conv_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
        skips.append(Res_conv_block(y, n_out // 4, fil, dil_rate, std, swtch, alpha_val, reg_val))
        y = Conv_down_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    # Attention block
    y = Res_conv_block(y, 256, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = Attention(y, num_heads, key_dim)
    y = Res_conv_block(y, 256, fil, dil_rate, std, swtch, alpha_val, reg_val)
    y = Attention(y, num_heads, key_dim)
    y = Res_conv_block(y, 256, fil, dil_rate, std, swtch, alpha_val, reg_val)

    # Decoding path
    for i, n_out in enumerate([256, 128, 64]):
        y = Conv_up_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)
        y = layers.Concatenate(axis=-1)([y, skips[-(i + 1)]])
        y = Res_conv_block(y, n_out, fil, dil_rate, std, swtch, alpha_val, reg_val)

    y = Res_conv_block(y, 32, fil, dil_rate, std, swtch, alpha_val, reg_val)

    y = con2d(y, 32, 1, dil_rate, std, 0, 0, reg_val)
    y = con2d(y, out_vars, 1, dil_rate, std, 0, 0, reg_val)

    return models.Model(inputs=[inp, y_st, y_noise], outputs=y)

# Discriminator architecture
def Dis(inp_lat, inp_lon, out_lat, out_lon, chnl, out_vars, n_out, fil, dil_rate, std, alpha_val, reg_val):

    inp = layers.Input(shape=(inp_lat, inp_lon, chnl))

    # Interpolate the input to target shape using bilinear interpolation
    y = tf.image.resize(inp, [out_lat, out_lon], method='bilinear')

    y_st = layers.Input(shape=(out_lat, out_lon, 2))

    y_inp = layers.Input(shape=(out_lat, out_lon, out_vars))

    y = layers.Concatenate(axis=-1)([y, y_st, y_inp])

    for i in range(5):
        if i == 0:
            x = con2d(y, n_out, fil, dil_rate, std, 1, alpha_val, reg_val)
        else:
            x = con2d(x, n_out, fil, dil_rate, std, 1, alpha_val, reg_val)
        x = pool2d(1, x, 2)
        n_out = n_out + n_out
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation=actv_swtch(1, alpha_val))(x)
    output = layers.Dense(out_vars)(x)

    return models.Model(inputs= [inp, y_st, y_inp], outputs=output)
