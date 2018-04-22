from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

slim = tf.contrib.slim


# functions for neural network layers
@slim.add_arg_scope
def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """strided 2-D convolution with 'REFLECT' padding.

    Args:
        inputs: A 4-D tensor of size [batch, height, width, channel]
        num_outputs: An integer, the number of output filters
        kernel_size: An int with the kernel_size of the filters
        stride: An integer, the output stride
        rate: An integer, rate for atrous convolution
        scope: Optional scope

    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channel] with
            the convolution output.
    """
    if kernel_size == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size=1, stride=stride,
                           rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
        inputs = tf.pad(inputs, paddings=paddings, mode="REFLECT")
        outputs = slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                              rate=rate, padding='VALID', scope=scope)
        return outputs


@slim.add_arg_scope
def conv2d_resize(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    """deconvolution alternatively with the conv2d_transpose, where we
    first resize the inputs, and then convolve the results, see
        http://distill.pub/2016/deconv-checkerboard/

    Args:
        inputs: A 4-D tensor of size [batch, height, width, channel]
        num_outputs: An integer, the number of output filters
        kernel_size: An int with the kernel_size of the filters
        stride: An integer, the output stride
        rate: An integer, rate for atrous convolution
        scope: Optional scope

    Returns:
        output: A 4-D tensor of size [batch, height_out, width_out, channel] with
            the convolution output.
    """
    if stride == 1:
        return conv2d_same(inputs, num_outputs, kernel_size,
                           stride=1, rate=rate, scope=scope)
    else:
        stride_larger_than_one = tf.greater(stride, 1)
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        new_height, new_width = tf.cond(
            stride_larger_than_one,
            lambda: (height*stride, width*stride),
            lambda: (height, width))
        inputs_resize = tf.image.resize_nearest_neighbor(inputs,
                                                         [new_height, new_width])
        outputs = conv2d_same(inputs_resize, num_outputs, kernel_size,
                              stride=1, rate=rate, scope=scope)
        return outputs


@slim.add_arg_scope
def lrelu(inputs, leak=0.2, scope=None):
    """customized leaky ReLU activation function
        https://github.com/tensorflow/tensorflow/issues/4079
    """
    with tf.variable_scope(scope, 'lrelu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * inputs + f2 * tf.abs(inputs)


@slim.add_arg_scope
def instance_norm(inputs, epsilon=1e-10):
    inst_mean, inst_var = tf.nn.moments(inputs, [1, 2], keep_dims=True)
    normalized_inputs = tf.div(
        tf.subtract(inputs, inst_mean), tf.sqrt(tf.add(inst_var, epsilon)))
    return normalized_inputs


@slim.add_arg_scope
def residual_unit_v0(inputs, depth, output_collections=None, scope=None):
    """Residual block version 0, the input and output has the same depth

    Args:
      inputs: a tensor of size [batch, height, width, channel]
      depth: the depth of the resnet unit output
      output_collections: collection to add the resnet unit output
      scope: optional variable_scope

    Returns:
      The resnet unit's output
    """
    with tf.variable_scope(scope, 'res_unit_v0', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = inputs
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], scope='shortcut')

        residual = conv2d_same(inputs, depth, 3, stride=1, scope='conv1')
        with slim.arg_scope([slim.conv2d], activation_fn=None):
            residual = conv2d_same(residual, depth, 3, stride=1, scope='conv2')

        output = tf.nn.relu(shortcut + residual)

        return slim.utils.collect_named_outputs(
            output_collections, sc.original_name_scope, output)


@slim.add_arg_scope
def residual_block_downsample(inputs, depth, stride,
                              normalizer_fn=slim.layer_norm,
                              activation_fn=tf.nn.relu,
                              outputs_collections=None, scope=None):
    """Residual block version 2 for downsampling, with preactivation

    Args:
        inputs: a tensor of size [batch, height, width, channel]
        depth: the depth of the resnet unit output
        stride: the stride of the residual block
        normalizer_fn: normalizer function for the residual block
        activation_fn: activation function for the residual block
        outputs_collections: collection to add the resnet unit output
        scope: optional variable_scope

    Returns:
        The resnet unit's output
    """
    with tf.variable_scope(scope, 'res_block_downsample', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=normalizer_fn,
                            activation_fn=activation_fn):
            # preactivate the inputs
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = normalizer_fn(inputs, activation_fn=activation_fn, scope='preact')
            if depth == depth_in:
                shortcut = subsample(inputs, stride, scope='shortcut')
            else:
                with slim.arg_scope([slim.conv2d],
                                    normalizer_fn=None, activation_fn=None):
                    shortcut = conv2d_same(preact, depth, 1,
                                           stride=stride, scope='shortcut')

            depth_botteneck = int(depth / 4)
            residual = slim.conv2d(preact, depth_botteneck, [1, 1],
                                   stride=1, scope='conv1')
            residual = conv2d_same(residual, depth_botteneck, 3,
                                   stride=stride, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1],
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None, scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(
                outputs_collections, sc.original_name_scope, output)


@slim.add_arg_scope
def residual_block_upsample(inputs, depth, stride,
                            normalizer_fn=slim.layer_norm,
                            activation_fn=tf.nn.relu,
                            outputs_collections=None, scope=None):
    """Residual block version 2 for upsampling, with preactivation

    Args:
        inputs: a tensor of size [batch, height, width, channel]
        depth: the depth of the resnet unit output
        stride: the stride of the residual block
        normalizer_fn: the normalizer function used in this block
        activation_fn: the activation function used in this block
        outputs_collections: collection to add the resnet unit output
        scope: optional variable_scope

    Returns:
        The resnet unit's output
    """
    with tf.variable_scope(scope, 'res_block_upsample', [inputs]) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=normalizer_fn,
                            activation_fn=activation_fn):
            # preactivate the inputs
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = normalizer_fn(inputs, activation_fn=activation_fn, scope='preact')
            if depth == depth_in:
                shortcut = upsample(inputs, stride, scope='shortcut')
            else:
                with slim.arg_scope([slim.conv2d],
                                    normalizer_fn=None, activation_fn=None):
                    shortcut = conv2d_resize(preact, depth, 1, stride=stride, scope='shortcut')

            # calculate the residuals
            depth_botteneck = int(depth / 4)
            residual = slim.conv2d(preact, depth_botteneck, [1, 1],
                                   stride=1, scope='conv1')
            residual = conv2d_resize(residual, depth_botteneck, 3,
                                     stride=stride, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1],
                                   stride=1, normalizer_fn=None,
                                   activation_fn=None, scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(
                outputs_collections, sc.original_name_scope, output)


def subsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def upsample(inputs, factor, scope=None):
    if factor == 1:
        return inputs
    else:
        factor_larger_than_one = tf.greater(factor, 1)
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        new_height, new_width = tf.cond(
            factor_larger_than_one,
            lambda: (height*factor, width*factor),
            lambda: (height, width))
        resized_inputs = tf.image.resize_nearest_neighbor(
            inputs, [new_height, new_width], name=scope)
        return resized_inputs


def adaptive_instance_normalization(content_feature, style_feature):
    """adaptively transform the content feature by inverse instance normalization
    based on the 2nd order statistics of the style feature
    """
    normalized_content_feature = instance_norm(content_feature)
    inst_mean, inst_var = tf.nn.moments(style_feature, [1, 2], keep_dims=True)
    return tf.sqrt(inst_var) * normalized_content_feature + inst_mean


def whitening_colorization_transform(content_features, style_features):
    """transform the content feature based on the whitening and colorization transform"""
    content_shape = tf.shape(content_features)
    style_shape = tf.shape(style_features)

    # get the unbiased content and style features
    content_features = tf.reshape(
        content_features, shape=(content_shape[0], -1, content_shape[3]))
    style_features = tf.reshape(
        style_features, shape=(style_shape[0], -1, style_shape[3]))

    # get the covariance matrices
    content_gram = tf.matmul(content_features, content_features, transpose_a=True)
    content_gram /= tf.reduce_prod(tf.cast(content_shape[1:], tf.float32))
    style_gram = tf.matmul(style_features, style_features, transpose_a=True)
    style_gram /= tf.reduce_prod(tf.cast(style_shape[1:], tf.float32))

    #################################
    # converting the feature spaces #
    #################################
    s_c, u_c, v_c = tf.svd(content_gram, compute_uv=True)
    s_c = tf.expand_dims(s_c, axis=1)
    s_s, u_s, v_s = tf.svd(style_gram, compute_uv=True)
    s_s = tf.expand_dims(s_s, axis=1)

    # normalized features
    normalized_features = tf.matmul(content_features, u_c)
    normalized_features = tf.multiply(normalized_features, 1.0/(tf.sqrt(s_c+1e-5)))
    normalized_features = tf.matmul(normalized_features, v_c, transpose_b=True)

    # colorized features
    # broadcasting the tensors for matrix multiplication
    content_batch = tf.shape(u_c)[0]
    style_batch = tf.shape(u_s)[0]
    batch_multiplier = tf.cast(content_batch/style_batch, tf.int32)
    u_s = tf.tile(u_s, multiples=tf.stack([batch_multiplier, 1, 1]))
    v_s = tf.tile(v_s, multiples=tf.stack([batch_multiplier, 1, 1]))
    colorized_features = tf.matmul(normalized_features, u_s)
    colorized_features = tf.multiply(colorized_features, tf.sqrt(s_s+1e-5))
    colorized_features = tf.matmul(colorized_features, v_s, transpose_b=True)

    # reshape the colorized features
    colorized_features = tf.reshape(colorized_features, shape=content_shape)
    return colorized_features
