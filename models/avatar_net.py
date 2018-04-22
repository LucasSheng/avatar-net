from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from models import losses
from models import network_ops
from models import vgg
from models import vgg_decoder
from models import preprocessing

slim = tf.contrib.slim

network_map = {
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
}


class AvatarNet(object):
    def __init__(self, options):
        self.training_image_size = options.get('training_image_size')
        self.content_size = options.get('content_size')
        self.style_size = options.get('style_size')

        # network architecture
        self.network_name = options.get('network_name')

        # the loss layers for content and style similarity
        self.style_loss_layers = options.get('style_loss_layers')

        # style decorator option
        # 'AdaIN': adaptive instance normalization
        # 'ZCA': zero-phase component analysis
        self.style_coding = options.get('style_coding')

        # window size for the patch swapper
        self.patch_size = options.get('patch_size')

        # training quantities
        self.content_weight = options.get('content_weight')
        self.recons_weight = options.get('recons_weight')
        self.tv_weight = options.get('tv_weight')
        self.weight_decay = options.get('weight_decay')

        # gather summaries and initialize the losses
        self.total_loss = 0.0
        self.recons_loss = None
        self.content_loss = None
        self.tv_loss = None

        # summary and training ops
        self.train_op = None
        self.summaries = None

    def transfer_styles(self,
                        inputs,
                        styles,
                        inter_weight=1.0,
                        intra_weights=(1,)):
        """transfer the content image by style images

        Args:
            inputs: input images [batch_size, height, width, channel]
            styles: a list of input styles, in default the size is 1
            inter_weight: the blending weight between the content and style
            intra_weights: a list of blending weights among the styles,
                in default it is (1,)

        Returns:
            outputs: the stylized images [batch_size, height, width, channel]
        """
        if not isinstance(styles, (list, tuple)):
            styles = [styles]

        if not isinstance(intra_weights, (list, tuple)):
            intra_weights = [intra_weights]

        # 1) extract the style features
        styles_features = []
        for style in styles:
            style_image_features = losses.extract_image_features(
                style, self.network_name)
            style_features = losses.compute_content_features(
                style_image_features, self.style_loss_layers)
            styles_features.append(style_features)

        # 2) content features
        inputs_image_features = losses.extract_image_features(
            inputs, self.network_name)
        inputs_features = losses.compute_content_features(
            inputs_image_features, self.style_loss_layers)

        # 3) style decorator
        # the applied content feature from the content input
        selected_layer = self.style_loss_layers[-1]
        hidden_feature = inputs_features[selected_layer]

        # applying the style decorator
        blended_feature = 0.0
        n = 0
        for style_features in styles_features:
            swapped_feature = style_decorator(
                hidden_feature,
                style_features[selected_layer],
                style_coding=self.style_coding,
                patch_size=self.patch_size)
            blended_feature += intra_weights[n] * swapped_feature
            n += 1
        blended_feature = (1 - inter_weight) * hidden_feature + inter_weight * blended_feature

        # 4) decode the hidden feature to the output image
        with slim.arg_scope(vgg_decoder.vgg_decoder_arg_scope()):
            outputs = vgg_decoder.vgg_multiple_combined_decoder(
                blended_feature,
                styles_features,
                intra_weights,
                fusion_fn=network_ops.adaptive_instance_normalization,
                network_name=self.network_name,
                starting_layer=selected_layer)
        return outputs

    def hierarchical_autoencoder(self, inputs, reuse=True):
        """hierarchical autoencoder for content reconstruction"""
        # extract the content features
        image_features = losses.extract_image_features(
            inputs, self.network_name)
        content_features = losses.compute_content_features(
            image_features, self.style_loss_layers)

        # the applied content feature for the decode network
        selected_layer = self.style_loss_layers[-1]
        hidden_feature = content_features[selected_layer]

        # decode the hidden feature to the output image
        with slim.arg_scope(vgg_decoder.vgg_decoder_arg_scope(self.weight_decay)):
            outputs = vgg_decoder.vgg_combined_decoder(
                hidden_feature,
                content_features,
                fusion_fn=network_ops.adaptive_instance_normalization,
                network_name=self.network_name,
                starting_layer=selected_layer,
                reuse=reuse)
        return outputs

    def build_train_graph(self, inputs):
        """build the training graph for the training of the hierarchical autoencoder"""
        outputs = self.hierarchical_autoencoder(inputs, reuse=False)
        outputs = preprocessing.batch_mean_image_subtraction(outputs)

        # summaries
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        ########################
        # construct the losses #
        ########################
        # 1) reconstruction loss
        if self.recons_weight > 0.0:
            recons_loss = tf.losses.mean_squared_error(
                inputs, outputs, weights=self.recons_weight, scope='recons_loss')
            self.recons_loss = recons_loss
            self.total_loss += recons_loss
            summaries.add(tf.summary.scalar('losses/recons_loss', recons_loss))

        # 2) content loss
        if self.content_weight > 0.0:
            outputs_image_features = losses.extract_image_features(
                outputs, self.network_name)
            outputs_content_features = losses.compute_content_features(
                outputs_image_features, self.style_loss_layers)

            inputs_image_features = losses.extract_image_features(
                inputs, self.network_name)
            inputs_content_features = losses.compute_content_features(
                inputs_image_features, self.style_loss_layers)

            content_loss = losses.compute_content_loss(
                outputs_content_features, inputs_content_features,
                content_loss_layers=self.style_loss_layers, weights=self.content_weight)
            self.content_loss = content_loss
            self.total_loss += content_loss
            summaries.add(tf.summary.scalar('losses/content_loss', content_loss))

        # 3) total variation loss
        if self.tv_weight > 0.0:
            tv_loss = losses.compute_total_variation_loss_l1(outputs, self.tv_weight)
            self.tv_loss = tv_loss
            self.total_loss += tv_loss
            summaries.add(tf.summary.scalar('losses/tv_loss', tv_loss))

        image_tiles = tf.concat([inputs, outputs], axis=2)
        image_tiles = preprocessing.batch_mean_image_summation(image_tiles)
        image_tiles = tf.cast(tf.clip_by_value(image_tiles, 0.0, 255.0), tf.uint8)
        summaries.add(tf.summary.image('image_comparison', image_tiles, max_outputs=8))

        self.summaries = summaries
        return self.total_loss

    def get_training_operations(self,
                                optimizer,
                                global_step,
                                variables_to_train=tf.trainable_variables()):
        # gather the variable summaries
        variables_summaries = []
        for var in variables_to_train:
            variables_summaries.append(tf.summary.histogram(var.op.name, var))
        variables_summaries = set(variables_summaries)

        # add the training operations
        train_ops = []
        grads_and_vars = optimizer.compute_gradients(
            self.total_loss, var_list=variables_to_train)
        train_op = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars,
            global_step=global_step)
        train_ops.append(train_op)

        self.summaries |= variables_summaries
        self.train_op = tf.group(*train_ops)
        return self.train_op


def style_decorator(content_features,
                    style_features,
                    style_coding='ZCA',
                    patch_size=3):
    """style decorator for high-level feature interaction

    Args:
        content_features: a tensor of size [batch_size, height, width, channel]
        style_features: a tensor of size [batch_size, height, width, channel]
        style_coding: projection and reconstruction method for style coding
        patch_size: a 0D tensor or int about the size of the patch
    """
    # feature projection
    projected_content_features, _, _ = \
        project_features(content_features, projection_module=style_coding)
    projected_style_features, style_kernels, mean_style_features = \
        project_features(style_features, projection_module=style_coding)

    # feature rearrangement
    rearranged_features = nearest_patch_swapping(
        projected_content_features, projected_style_features, patch_size=patch_size)

    # feature reconstruction
    reconstructed_features = reconstruct_features(
        rearranged_features,
        style_kernels,
        mean_style_features,
        reconstruction_module=style_coding)
    return reconstructed_features


def project_features(features, projection_module='ZCA'):
    if projection_module is 'ZCA':
        projected_features, feature_kernels, mean_features = zca_normalization(features)
    elif projection_module is 'AdaIN':
        projected_features, feature_kernels, mean_features = adain_normalization(features)
    else:
        projected_features = features
        feature_kernels, mean_features = None, None
    return projected_features, feature_kernels, mean_features


def reconstruct_features(projected_features,
                         feature_kernels,
                         mean_features,
                         reconstruction_module='ZCA'):
    if reconstruction_module is 'ZCA':
        reconstructed_features = zca_colorization(
            projected_features, feature_kernels, mean_features)
    elif reconstruction_module is 'AdaIN':
        reconstructed_features = adain_colorization(
            projected_features, feature_kernels, mean_features)
    else:
        reconstructed_features = projected_features
    return reconstructed_features


def nearest_patch_swapping(content_features, style_features, patch_size=3):
    # channels for both the content and style, must be the same
    c_shape = tf.shape(content_features)
    s_shape = tf.shape(style_features)
    channel_assertion = tf.Assert(
        tf.equal(c_shape[3], s_shape[3]), ['number of channels  must be the same'])

    with tf.control_dependencies([channel_assertion]):
        # spatial shapes for style and content features
        c_height, c_width, c_channel = c_shape[1], c_shape[2], c_shape[3]

        # convert the style features into convolutional kernels
        style_kernels = tf.extract_image_patches(
            style_features, ksizes=[1, patch_size, patch_size, 1],
            strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
        style_kernels = tf.squeeze(style_kernels, axis=0)
        style_kernels = tf.transpose(style_kernels, perm=[2, 0, 1])

        # gather the conv and deconv kernels
        v_height, v_width = style_kernels.get_shape().as_list()[1:3]
        deconv_kernels = tf.reshape(
            style_kernels, shape=(patch_size, patch_size, c_channel, v_height*v_width))

        kernels_norm = tf.norm(style_kernels, axis=0, keep_dims=True)
        kernels_norm = tf.reshape(kernels_norm, shape=(1, 1, 1, v_height*v_width))

        # calculate the normalization factor
        mask = tf.ones((c_height, c_width), tf.float32)
        fullmask = tf.zeros((c_height+patch_size-1, c_width+patch_size-1), tf.float32)
        for x in range(patch_size):
            for y in range(patch_size):
                paddings = [[x, patch_size-x-1], [y, patch_size-y-1]]
                padded_mask = tf.pad(mask, paddings=paddings, mode="CONSTANT")
                fullmask += padded_mask
        pad_width = int((patch_size-1)/2)
        deconv_norm = tf.slice(fullmask, [pad_width, pad_width], [c_height, c_width])
        deconv_norm = tf.reshape(deconv_norm, shape=(1, c_height, c_width, 1))

        ########################
        # starting convolution #
        ########################
        # padding operation
        pad_total = patch_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

        # convolutional operations
        net = tf.pad(content_features, paddings=paddings, mode="REFLECT")
        net = tf.nn.conv2d(
            net,
            tf.div(deconv_kernels, kernels_norm+1e-7),
            strides=[1, 1, 1, 1],
            padding='VALID')
        # find the maximum locations
        best_match_ids = tf.argmax(net, axis=3)
        best_match_ids = tf.cast(
            tf.one_hot(best_match_ids, depth=v_height*v_width), dtype=tf.float32)

        # find the patches and warping the output
        unnormalized_output = tf.nn.conv2d_transpose(
            value=best_match_ids,
            filter=deconv_kernels,
            output_shape=(c_shape[0], c_height+pad_total, c_width+pad_total, c_channel),
            strides=[1, 1, 1, 1],
            padding='VALID')
        unnormalized_output = tf.slice(unnormalized_output, [0, pad_beg, pad_beg, 0], c_shape)
        output = tf.div(unnormalized_output, deconv_norm)
        output = tf.reshape(output, shape=c_shape)

        # output the swapped feature maps
        return output


def zca_normalization(features):
    shape = tf.shape(features)

    # reshape the features to orderless feature vectors
    mean_features = tf.reduce_mean(features, axis=[1, 2], keep_dims=True)
    unbiased_features = tf.reshape(features - mean_features, shape=(shape[0], -1, shape[3]))

    # get the covariance matrix
    gram = tf.matmul(unbiased_features, unbiased_features, transpose_a=True)
    gram /= tf.reduce_prod(tf.cast(shape[1:3], tf.float32))

    # converting the feature spaces
    s, u, v = tf.svd(gram, compute_uv=True)
    s = tf.expand_dims(s, axis=1)  # let it be active in the last dimension

    # get the effective singular values
    valid_index = tf.cast(s > 0.00001, dtype=tf.float32)
    s_effective = tf.maximum(s, 0.00001)
    sqrt_s_effective = tf.sqrt(s_effective) * valid_index
    sqrt_inv_s_effective = tf.sqrt(1.0/s_effective) * valid_index

    # colorization functions
    colorization_kernel = tf.matmul(tf.multiply(u, sqrt_s_effective), v, transpose_b=True)

    # normalized features
    normalized_features = tf.matmul(unbiased_features, u)
    normalized_features = tf.multiply(normalized_features, sqrt_inv_s_effective)
    normalized_features = tf.matmul(normalized_features, v, transpose_b=True)
    normalized_features = tf.reshape(normalized_features, shape=shape)

    return normalized_features, colorization_kernel, mean_features


def zca_colorization(normalized_features, colorization_kernel, mean_features):
    # broadcasting the tensors for matrix multiplication
    shape = tf.shape(normalized_features)
    normalized_features = tf.reshape(
        normalized_features, shape=(shape[0], -1, shape[3]))
    colorized_features = tf.matmul(normalized_features, colorization_kernel)
    colorized_features = tf.reshape(colorized_features, shape=shape) + mean_features
    return colorized_features


def adain_normalization(features):
    epsilon = 1e-7
    mean_features, colorization_kernels = tf.nn.moments(features, [1, 2], keep_dims=True)
    normalized_features = tf.div(
        tf.subtract(features, mean_features), tf.sqrt(tf.add(colorization_kernels, epsilon)))
    return normalized_features, colorization_kernels, mean_features


def adain_colorization(normalized_features, colorization_kernels, mean_features):
    return tf.sqrt(colorization_kernels) * normalized_features + mean_features
