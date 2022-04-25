import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, Lambda, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_normal

import tensorflow.keras.backend as K

def abpn(scale=3, in_channels=3, num_fea=28, m=4, out_channels=3):
    inp = Input(shape=(None, None, 3)) 
    upsample_func = Lambda(lambda x_list: tf.concat(x_list, axis=3))
    #upsampled_inp = upsample_func([inp, inp, inp, inp, inp, inp, inp, inp, inp])
    upsampled_inp = upsample_func([inp for x in range(scale**2)])

    # Feature extraction
    x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(inp)

    for i in range(m):
        x = Conv2D(num_fea, 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)

    # Pixel-Shuffle
    x = Conv2D(out_channels*(scale**2), 3, padding='same', activation='relu', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Conv2D(out_channels*(scale**2), 3, padding='same', kernel_initializer=glorot_normal(), bias_initializer='zeros')(x)
    x = Add()([upsampled_inp, x])
    
    depth_to_space = Lambda(lambda x: tf.nn.depth_to_space(x, scale))
    out = depth_to_space(x)
    clip_func = Lambda(lambda x: K.clip(x, 0., 255.))
    # clip_func = Lambda(lambda x: tf.clip_by_value(x, 0., 255.))
    out = clip_func(out)
    
    return Model(inputs=inp, outputs=out, name='abpn')
