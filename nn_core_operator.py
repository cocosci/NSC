from loss_terms_and_measures import *
# from constants import *
import tensorflow as tf


def conv1d(inputs, num_filters, filter_size, padding='SAME', dilation_rate=1, strides=1, activation=tf.nn.tanh):
    out = tf.compat.v1.layers.conv1d(
            inputs=inputs, filters=num_filters, padding=padding,
            kernel_size=filter_size,
            activation=activation,
            dilation_rate=dilation_rate,
            strides=strides,
            data_format='channels_last')
    return out


def conv1d_depth(inputs, num_filters, filter_size, padding='SAME', dilation_rate=1, strides=1, activation=tf.nn.tanh):
    out = tf.keras.layers.SeparableConv1D(filters=num_filters, padding=padding, kernel_size=filter_size,
                                          activation=activation, dilation_rate=dilation_rate, strides=strides,
                                          data_format='channels_last')(inputs)
    return out


def activation_func(_x):
    #alphas = tf.Variable(tf.ones(shape=_x.get_shape()[-1]), name='prelu_alpha'+str(np.random.randint(1000000, 2000000)))
    #pos = tf.nn.relu(_x)
    #neg = alphas * (_x - abs(_x)) * 0.5
    #return pos + neg
    #return tf.nn.elu(_x)
    return tf.nn.leaky_relu(_x)
    # return tf.nn.relu(_x)


def batch_norm(_x, training):
    # return tf.cond(
    #     training,
    #     lambda: tf.keras.layers.BatchNormalization()(_x, training=True),
    #     lambda: tf.keras.layers.BatchNormalization()(_x, training=False),
    # )
    return _x
    # return tf.compat.v1.layers.batch_normalization(inputs=_x, axis=1, training=True)  # axis would be 1 for
    # theano-type dim.


def change_channel(the_input,
                   wide_layer=30,
                   the_channel=1,
                   kernel_size=9,
                   dilation_rate=1,
                   strides=1,
                   activation=None):
    compressed_bit = conv1d(the_input, the_channel, filter_size=kernel_size, padding='SAME', dilation_rate=1,
                            strides=strides, activation=activation)
    return compressed_bit


def the_bottleneck(the_input, wide_layer=30, narrow_layer=10, non_dilated_neck_kernel_size=9,
                   dilated_neck_kernel_size=9, dilation_rate=1, is_last_flat=False):
    compressed_bit_2 = conv1d(the_input, narrow_layer,
                              filter_size=non_dilated_neck_kernel_size,
                              padding='SAME',
                              dilation_rate=1, activation=None)
    compressed_bit_2 = activation_func(compressed_bit_2)

    compressed_bit_2 = conv1d(compressed_bit_2, narrow_layer,
                              filter_size=dilated_neck_kernel_size,
                              padding='SAME',
                              dilation_rate=dilation_rate, activation=None)
    compressed_bit_2 = activation_func(compressed_bit_2)

    compressed_bit_2 = conv1d(compressed_bit_2, wide_layer,
                              filter_size=non_dilated_neck_kernel_size,
                              padding='SAME',
                              dilation_rate=1, activation=None)
    # compressed_bit_2 = activation_func(compressed_bit_2)
    if not is_last_flat:
        return activation_func(compressed_bit_2 + the_input)
    else:
        return compressed_bit_2 + the_input


def gated_bottleneck(the_input, wide_layer=30, narrow_layer=10, non_dilated_neck_kernel_size=9, dilated_neck_kernel_size = 9, dilation_rate = 1, is_last_flat = False, the_share=False):
    compressed_bit_2 = conv1d(the_input, narrow_layer,
                              filter_size=1,
                              padding='SAME',
                             dilation_rate=1, activation=None)
    # compressed_bit_2 = batch_norm(compressed_bit_2, the_share)
    compressed_bit_2 = activation_func(compressed_bit_2)
    # gated linear unit
    compressed_bit_3_left = conv1d(compressed_bit_2, narrow_layer,
                                   # filter_size=dilated_neck_kernel_size,
                                   filter_size=15,
                                   padding='SAME',
                                   dilation_rate=dilation_rate, activation=None)
    compressed_bit_3_right = conv1d(compressed_bit_2, narrow_layer,
                                    # filter_size=dilated_neck_kernel_size,
                                    filter_size=15,
                                    padding='SAME',
                                   dilation_rate=dilation_rate, activation=tf.nn.tanh)
    compressed_bit_3 = tf.multiply(compressed_bit_3_left, compressed_bit_3_right)

    compressed_bit_2 = conv1d(compressed_bit_3, wide_layer, filter_size=non_dilated_neck_kernel_size, padding='SAME',
                             dilation_rate=1, activation=None)

    # compressed_bit_2 = tf.compat.v1.layers.batch_normalization(compressed_bit_2)
    # compressed_bit_2 = batch_norm(compressed_bit_2, the_share)

    # compressed_bit_2 = activation_func(compressed_bit_2)
    if not is_last_flat:
        return activation_func(compressed_bit_2 + the_input)
    else:
        return compressed_bit_2 + the_input


def gated_bottleneck_decoder(the_input, wide_layer=30, narrow_layer=10, non_dilated_neck_kernel_size=9, dilated_neck_kernel_size = 9, dilation_rate = 1, is_last_flat = False, the_share=False):
    compressed_bit_2 = conv1d(the_input, narrow_layer, filter_size=1, padding='SAME',
                                    dilation_rate=1, activation=None)
    # compressed_bit_2 = batch_norm(compressed_bit_2, the_share)
    compressed_bit_2 = activation_func(compressed_bit_2)
    # gated linear unit
    compressed_bit_3_left = conv1d(compressed_bit_2, narrow_layer, filter_size=dilated_neck_kernel_size, padding='SAME',
                                         dilation_rate=dilation_rate, activation=None)
    compressed_bit_3_right = conv1d(compressed_bit_2, narrow_layer, filter_size=dilated_neck_kernel_size, padding='SAME',
                                          dilation_rate=dilation_rate, activation=tf.nn.tanh)
    compressed_bit_3 = tf.multiply(compressed_bit_3_left, compressed_bit_3_right)

    compressed_bit_2 = conv1d_depth(compressed_bit_3, wide_layer, filter_size=non_dilated_neck_kernel_size, padding='SAME',
                                    dilation_rate=1, activation=None)

    # compressed_bit_2 = tf.compat.v1.layers.batch_normalization(compressed_bit_2)
    # compressed_bit_2 = batch_norm(compressed_bit_2, the_share)

    # compressed_bit_2 = activation_func(compressed_bit_2)
    if not is_last_flat:
        return activation_func(compressed_bit_2 + the_input)
    else:
        return compressed_bit_2 + the_input


def scalar_softmax_quantization(floating_code, alpha, bins, is_quan_on, the_share, code_length, num_kmean_kernels):
    bins_expand = tf.expand_dims(bins, 1)
    bins_expand = tf.reshape(bins_expand, (1, 1, -1))
    dist = tf.abs(floating_code - bins_expand)
    bottle_neck_size = floating_code.shape[1]
    print(bins_expand.shape, floating_code.shape, dist.shape)
    soft_assignment = tf.nn.softmax(tf.multiply(alpha, dist))  # frame_length * 256
    soft_assignment_3d = soft_assignment
    # input()
    hard_assignment = tf.reshape(tf.one_hot(tf.nn.top_k(soft_assignment).indices, num_kmean_kernels),
                                 (-1, code_length, num_kmean_kernels))
    print('hard_assignment', hard_assignment.shape)  # lpc ? 16 64
    print('soft_assignment', soft_assignment.shape)  # lpc <unknown>

    soft_assignment = tf.cond(
        the_share,
        lambda: soft_assignment,
        lambda: hard_assignment,
    )

    bit_code = tf.reshape(tf.matmul(soft_assignment, tf.expand_dims(bins, 1)), (-1, bottle_neck_size, 1))
    # is_quan_on = (is_quan_on==1)
    bit_code = tf.cast((1 - is_quan_on) * floating_code, tf.float32) + tf.cast(is_quan_on * bit_code, tf.float32)
    bit_code = tf.reshape(bit_code, (-1, bottle_neck_size, 1))
    return soft_assignment_3d, bit_code


def vector_softmax_quantization(floating_code, alpha, bins, is_quan_on, is_share, top_k, code_len):
    print(floating_code.shape, 'floating_code')
    assert floating_code.shape[1]%code_len == 0
    floating_code = tf.reshape(floating_code, (-1, code_len, int(floating_code.shape[1] / code_len)))
    print(bins.shape, 'codebook shape')
    floating_code = tf.expand_dims(floating_code, 2)
    print(floating_code.shape, bins.shape, 'floating_code')  # (None, 256, 1, 1) (32, 1) floating_code
    dist = vec_l2norm_tf(floating_code - bins)
    print(dist.shape, 'dist.shape')  # (None, 256, 32) dist.shape

    soft_assignment = tf.nn.softmax(tf.multiply(alpha, dist))
    hard_assignment = tf.reshape(tf.one_hot(tf.nn.top_k(soft_assignment).indices, code_book_size_val),
                                 (-1, code_len, code_book_size_val))
    soft_code = tf.reshape(tf.matmul(soft_assignment, bins), (-1, bottle_neck_size, 1))
    #soft_code = tf.cast((1 - is_quan_on) * floating_code, tf.float32) + tf.cast(is_quan_on * soft_code, tf.float32)
    #soft_code = tf.reshape(soft_code, (-1, bottle_neck_size, 1))
    print('soft_code.shape', soft_code.shape)  # (None, 256, 1)
    print('soft_assignment_vq', soft_assignment.shape)
    print('hard_assignment_vq', hard_assignment.shape)
    top_prob, top_indices = (tf.nn.top_k(soft_assignment, k=1))  ###

    selected_bins = tf.gather(bins, top_indices)[:, :, 0, :]
    vq_decoded = tf.reshape(selected_bins, (-1, bottle_neck_size, 1))

    return tf.cond(
        is_share,
        lambda: [soft_assignment, soft_code], # tf.random.normal(shape=tf.shape(non_vq_floating_code), mean=0.0, stddev=0.01, dtype=tf.float32) + non_vq_floating_code,  # no
        lambda: [hard_assignment, vq_decoded],  # vq_decoded,  #[?, 256, 0]
    )
