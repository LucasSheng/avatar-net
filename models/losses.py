from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models import vgg

slim = tf.contrib.slim

network_map = {
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
}


def compute_gram_matrix(feature):
    """compute the gram matrix for a layer of feature

    the gram matrix is normalized with respect to the samples and
        the dimensions of the input features

    """
    shape = tf.shape(feature)
    feature_size = tf.reduce_prod(shape[1:])
    vectorized_feature = tf.reshape(
        feature, [shape[0], -1, shape[3]])
    gram_matrix = tf.matmul(
        vectorized_feature, vectorized_feature, transpose_a=True)
    gram_matrix /= tf.to_float(feature_size)
    return gram_matrix


def compute_sufficient_statistics(feature):
    """compute the gram matrix for a layer of feature"""
    mean_feature, var_feature = tf.nn.moments(feature, [1, 2], keep_dims=True)
    std_feature = tf.sqrt(var_feature)
    sufficient_statistics = tf.concat([mean_feature, std_feature], axis=3)
    return sufficient_statistics


def compute_content_features(features, content_loss_layers):
    """compute the content features from the end_point dict"""
    content_features = {}
    instance_label = features.keys()[0]
    instance_label = instance_label[:-14]  # TODO: ugly code, need fix
    for layer in content_loss_layers:
        content_features[layer] = features[instance_label + '/' + layer]
    return content_features


def compute_style_features(features, style_loss_layers):
    """compute the style features from the end_point dict"""
    style_features = {}
    instance_label = features.keys()[0]
    instance_label = instance_label[:-14]  # TODO: ugly code, need fix
    for layer in style_loss_layers:
        style_features[layer] = compute_gram_matrix(
            features[instance_label + '/' + layer])
    return style_features


def compute_approximate_style_features(features, style_loss_layers):
    style_features = {}
    instance_label = features.keys()[0].split('/')[:-2]
    for layer in style_loss_layers:
        style_features[layer] = compute_sufficient_statistics(
            features[instance_label + '/' + layer])
    return style_features


def extract_image_features(inputs, network_name, reuse=True):
    """compute the dict of layer-wise image features from a given list of networks

    Args:
      inputs: the inputs image should be normalized between [-127.5, 127.5]
      network_name: the network name for the perceptual loss
      reuse: whether to reuse the parameters

    Returns:
      end_points: a dict for the image features of the inputs
    """
    with slim.arg_scope(vgg.vgg_arg_scope()):
        _, end_points = network_map[network_name](
            inputs, spatial_squeeze=False, is_training=False, reuse=reuse)
    return end_points


def compute_content_and_style_features(inputs,
                                       network_name,
                                       content_loss_layers,
                                       style_loss_layers):
    """compute the content and style features from normalized image

    Args:
      inputs: input tensor of size [batch, height, width, channel]
      network_name: a string of the network name
      content_loss_layers: a dict about the layers for the content loss
      style_loss_layers: a dict about the layers for the style loss

    Returns:
      a dict of the features of the inputs
    """
    end_points = extract_image_features(inputs, network_name)

    content_features = compute_content_features(end_points, content_loss_layers)
    style_features = compute_style_features(end_points, style_loss_layers)

    return content_features, style_features


def compute_content_loss(content_features, target_features,
                         content_loss_layers, weights=1, scope=None):
    """compute the content loss

    Args:
      content_features: a dict of the features of the input image
      target_features: a dict of the features of the output image
      content_loss_layers: a dict about the layers for the content loss
      weights: the weights for this loss
      scope: optional scope

    Returns:
      the content loss
    """
    with tf.variable_scope(scope, 'content_loss', [content_features, target_features]):
        content_loss = 0
        for layer in content_loss_layers:
            content_feature = content_features[layer]
            target_feature = target_features[layer]
            content_loss += tf.losses.mean_squared_error(
                target_feature, content_feature, weights=weights, scope=layer)
        return content_loss


def compute_style_loss(style_features, target_features,
                       style_loss_layers, weights=1, scope=None):
    """compute the style loss

    Args:
        style_features: a dict of the Gram matrices of the style image
        target_features: a dict of the Gram matrices of the target image
        style_loss_layers: a dict of layers of features for the style loss
        weights: the weights for this loss
        scope: optional scope

    Returns:
        the style loss
    """
    with tf.variable_scope(scope, 'style_loss', [style_features, target_features]):
        style_loss = 0
        for layer in style_loss_layers:
            style_feature = style_features[layer]
            target_feature = target_features[layer]
            style_loss += tf.losses.mean_squared_error(
                style_feature, target_feature, weights=weights, scope=layer)
    return style_loss


def compute_approximate_style_loss(style_features, target_features,
                                   style_loss_layers, scope=None):
    """compute the approximate style loss

    Args:
        style_features: a dict of the sufficient statistics of the
            feature maps of the style image
        target_features: a dict of the sufficient statistics of the
            feature maps of the target image
        style_loss_layers: a dict of layers of features for the style loss
        scope: optional scope

    Returns:
        the style loss
    """
    with tf.variable_scope(scope, 'approximated_style_loss', [style_features, target_features]):
        style_loss = 0
        for layer in style_loss_layers:
            style_feature = style_features[layer]
            target_feature = target_features[layer]
            # we only normalize with respect to the number of channel
            style_loss_per_layer = tf.reduce_sum(tf.square(style_feature-target_feature), axis=[1, 2, 3])
            style_loss += tf.reduce_mean(style_loss_per_layer)
    return style_loss


def compute_total_variation_loss_l2(inputs, weights=1, scope=None):
    """compute the total variation loss"""
    inputs_shape = tf.shape(inputs)
    height = inputs_shape[1]
    width = inputs_shape[2]

    with tf.variable_scope(scope, 'total_variation_loss', [inputs]):
        loss_y = tf.losses.mean_squared_error(
            tf.slice(inputs, [0, 0, 0, 0], [-1, height-1, -1, -1]),
            tf.slice(inputs, [0, 1, 0, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_y')
        loss_x = tf.losses.mean_squared_error(
            tf.slice(inputs, [0, 0, 0, 0], [-1, -1, width-1, -1]),
            tf.slice(inputs, [0, 0, 1, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_x')
        loss = loss_y + loss_x
        return loss


def compute_total_variation_loss_l1(inputs, weights=1, scope=None):
    """compute the total variation loss L1 norm"""
    inputs_shape = tf.shape(inputs)
    height = inputs_shape[1]
    width = inputs_shape[2]

    with tf.variable_scope(scope, 'total_variation_loss', [inputs]):
        loss_y = tf.losses.absolute_difference(
            tf.slice(inputs, [0, 0, 0, 0], [-1, height-1, -1, -1]),
            tf.slice(inputs, [0, 1, 0, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_y')
        loss_x = tf.losses.absolute_difference(
            tf.slice(inputs, [0, 0, 0, 0], [-1, -1, width-1, -1]),
            tf.slice(inputs, [0, 0, 1, 0], [-1, -1, -1, -1]),
            weights=weights,
            scope='loss_x')
        loss = loss_y + loss_x
        return loss
