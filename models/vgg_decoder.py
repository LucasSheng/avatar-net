from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from models import network_ops

slim = tf.contrib.slim

vgg_19_decoder_architecture = [
    ('conv5/conv5_4', ('c', 512, 3)),
    ('conv5/conv5_3', ('c', 512, 3)),
    ('conv5/conv5_2', ('c', 512, 3)),
    ('conv5/conv5_1', ('c', 512, 3)),
    ('conv4/conv4_4', ('uc', 512, 3)),
    ('conv4/conv4_3', ('c', 512, 3)),
    ('conv4/conv4_2', ('c', 512, 3)),
    ('conv4/conv4_1', ('c', 256, 3)),
    ('conv3/conv3_4', ('uc', 256, 3)),
    ('conv3/conv3_3', ('c', 256, 3)),
    ('conv3/conv3_2', ('c', 256, 3)),
    ('conv3/conv3_1', ('c', 128, 3)),
    ('conv2/conv2_2', ('uc', 128, 3)),
    ('conv2/conv2_1', ('c', 64, 3)),
    ('conv1/conv1_2', ('uc', 64, 3)),
    ('conv1/conv1_1', ('c', 64, 3)),
]

vgg_16_decoder_architecture = [
    ('conv5/conv5_3', ('c', 512, 3)),
    ('conv5/conv5_2', ('c', 512, 3)),
    ('conv5/conv5_1', ('c', 512, 3)),
    ('conv4/conv4_3', ('uc', 512, 3)),
    ('conv4/conv4_2', ('c', 512, 3)),
    ('conv4/conv4_1', ('c', 256, 3)),
    ('conv3/conv3_3', ('uc', 256, 3)),
    ('conv3/conv3_2', ('c', 256, 3)),
    ('conv3/conv3_1', ('c', 128, 3)),
    ('conv2/conv2_2', ('uc', 128, 3)),
    ('conv2/conv2_1', ('c', 64, 3)),
    ('conv1/conv1_2', ('uc', 64, 3)),
    ('conv1/conv1_1', ('c', 64, 3)),
]

network_map = {
    'vgg_19': vgg_19_decoder_architecture,
    'vgg_16': vgg_16_decoder_architecture,
}


def vgg_decoder_arg_scope(weight_decay=0.0005):
    with slim.arg_scope(
            [slim.conv2d],
            padding='SAME',
            activation_fn=tf.nn.relu,
            normalizer_fn=None,
            weights_initializer=slim.xavier_initializer(uniform=False),
            weights_regularizer=slim.l2_regularizer(weight_decay)) as arg_sc:
        return arg_sc


def vgg_decoder(inputs,
                network_name='vgg_16',
                starting_layer='conv1/conv1_1',
                reuse=False,
                scope=None):
    """construct the decoder network for the vgg models

    Args:
        inputs: input features [batch_size, height, width, channel]
        network_name: the type of the network, default is vgg_16
        starting_layer: the starting reflectance layer, default is 'conv1/conv1_1'
        reuse: (optional) whether to reuse the network
        scope: (optional) the scope of the network

    Returns:
        outputs: the decoded feature maps
    """
    with tf.variable_scope(scope, 'image_decoder', reuse=reuse):
        # gather the output with identity mapping
        net = tf.identity(inputs)

        # starting inferring the network
        is_active = False
        for layer, layer_struct in network_map[network_name]:
            if layer == starting_layer:
                is_active = True
            if is_active:
                conv_type, num_outputs, kernel_size = layer_struct
                if conv_type == 'c':
                    net = network_ops.conv2d_same(net, num_outputs, kernel_size, 1, scope=layer)
                elif conv_type == 'uc':
                    net = network_ops.conv2d_resize(net, num_outputs, kernel_size, 2, scope=layer)
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, activation_fn=tf.tanh):
            outputs = network_ops.conv2d_same(net, 3, 7, 1, scope='output')
        return outputs * 150.0 + 127.5


def vgg_combined_decoder(inputs,
                         additional_features,
                         fusion_fn=None,
                         network_name='vgg_16',
                         starting_layer='conv1/conv1_1',
                         reuse=False,
                         scope=None):
    """construct the decoder network with additional feature combination

    Args:
        inputs: input features [batch_size, height, width, channel]
        additional_features: a dict contains the additional features
        fusion_fn: the fusion function to combine features
        network_name: the type of the network, default is vgg_16
        starting_layer: the starting reflectance layer, default is 'conv1/conv1_1'
        reuse: (optional) whether to reuse the network
        scope: (optional) the scope of the network

    Returns:
        outputs: the decoded feature maps
    """
    with tf.variable_scope(scope, 'combined_decoder', reuse=reuse):
        # gather the output with identity mapping
        net = tf.identity(inputs)

        # starting inferring the network
        is_active = False
        for layer, layer_struct in network_map[network_name]:
            if layer == starting_layer:
                is_active = True
            if is_active:
                conv_type, num_outputs, kernel_size = layer_struct

                # combine the feature
                add_feature = additional_features.get(layer)
                if add_feature is not None and layer != starting_layer:
                    net = fusion_fn(net, add_feature)

                if conv_type == 'c':
                    net = network_ops.conv2d_same(net, num_outputs, kernel_size, 1, scope=layer)
                elif conv_type == 'uc':
                    net = network_ops.conv2d_resize(net, num_outputs, kernel_size, 2, scope=layer)
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, activation_fn=None):
            outputs = network_ops.conv2d_same(net, 3, 7, 1, scope='output')
        return outputs + 127.5


def vgg_multiple_combined_decoder(inputs,
                                  additional_features,
                                  blending_weights,
                                  fusion_fn=None,
                                  network_name='vgg_16',
                                  starting_layer='conv1/conv1_1',
                                  reuse=False,
                                  scope=None):
    """construct the decoder network with additional feature combination

    Args:
        inputs: input features [batch_size, height, width, channel]
        additional_features: a dict contains the additional features
        blending_weights: the list of weights used for feature blending
        fusion_fn: the fusion function to combine features
        network_name: the type of the network, default is vgg_16
        starting_layer: the starting reflectance layer, default is 'conv1/conv1_1'
        reuse: (optional) whether to reuse the network
        scope: (optional) the scope of the network

    Returns:
        outputs: the decoded feature maps
    """
    with tf.variable_scope(scope, 'combined_decoder', reuse=reuse):
        # gather the output with identity mapping
        net = tf.identity(inputs)

        # starting inferring the network
        is_active = False
        for layer, layer_struct in network_map[network_name]:
            if layer == starting_layer:
                is_active = True
            if is_active:
                conv_type, num_outputs, kernel_size = layer_struct

                # combine the feature
                add_feature = additional_features[0].get(layer)
                if add_feature is not None and layer != starting_layer:
                    # fuse multiple styles
                    n = 0
                    layer_output = 0.0
                    for additional_feature in additional_features:
                        additional_layer_feature = additional_feature.get(layer)
                        fused_layer_feature = fusion_fn(net, additional_layer_feature)
                        layer_output += blending_weights[n] * fused_layer_feature
                        n += 1
                    net = layer_output

                if conv_type == 'c':
                    net = network_ops.conv2d_same(net, num_outputs, kernel_size, 1, scope=layer)
                elif conv_type == 'uc':
                    net = network_ops.conv2d_resize(net, num_outputs, kernel_size, 2, scope=layer)
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, activation_fn=None):
            outputs = network_ops.conv2d_same(net, 3, 7, 1, scope='output')
        return outputs + 127.5
