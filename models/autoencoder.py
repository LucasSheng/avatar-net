from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from models import losses
from models import preprocessing
from models import vgg
from models import vgg_decoder

slim = tf.contrib.slim

network_map = {
    'vgg_16': vgg.vgg_16,
    'vgg_19': vgg.vgg_19,
}


class AutoEncoder(object):
    def __init__(self, options):
        self.weight_decay = options.get('weight_decay')

        self.default_size = options.get('default_size')
        self.content_size = options.get('content_size')

        # network architecture
        self.network_name = options.get('network_name')

        # the loss layers for content and style similarity
        self.content_layers = options.get('content_layers')

        # the weights for the losses when trains the invertible network
        self.content_weight = options.get('content_weight')
        self.recons_weight = options.get('recons_weight')
        self.tv_weight = options.get('tv_weight')

        # gather the summaries and initialize the losses
        self.summaries = None
        self.total_loss = 0.0
        self.recons_loss = {}
        self.content_loss = {}
        self.tv_loss = {}
        self.train_op = None

    def auto_encoder(self, inputs, content_layer=2, reuse=True):
        # extract the content features
        image_features = losses.extract_image_features(inputs, self.network_name)
        content_features = losses.compute_content_features(image_features, self.content_layers)

        # used content feature
        selected_layer = self.content_layers[content_layer]
        content_feature = content_features[selected_layer]
        input_content_features = {selected_layer: content_feature}

        # reconstruct the images
        with slim.arg_scope(vgg_decoder.vgg_decoder_arg_scope(self.weight_decay)):
            outputs = vgg_decoder.vgg_decoder(
                content_feature,
                self.network_name,
                selected_layer,
                reuse=reuse,
                scope='decoder_%d' % content_layer)
        return outputs, input_content_features

    def build_train_graph(self, inputs):
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        for i in range(len(self.content_layers)):
            # skip some networks
            if i < 3:
                continue

            selected_layer = self.content_layers[i]

            outputs, inputs_content_features = self.auto_encoder(
                inputs, content_layer=i, reuse=False)
            outputs = preprocessing.batch_mean_image_subtraction(outputs)

            ########################
            # construct the losses #
            ########################
            # 1) reconstruction loss
            recons_loss = tf.losses.mean_squared_error(
                inputs, outputs, scope='recons_loss/decoder_%d' % i)
            self.recons_loss[selected_layer] = recons_loss
            self.total_loss += self.recons_weight * recons_loss
            summaries.add(tf.summary.scalar(
                'recons_loss/decoder_%d' % i, recons_loss))

            # 2) content loss
            outputs_image_features = losses.extract_image_features(
                outputs, self.network_name)
            outputs_content_features = losses.compute_content_features(
                outputs_image_features, [selected_layer])
            content_loss = losses.compute_content_loss(
                outputs_content_features, inputs_content_features, [selected_layer])
            self.content_loss[selected_layer] = content_loss
            self.total_loss += self.content_weight * content_loss
            summaries.add(tf.summary.scalar(
                'content_loss/decoder_%d' % i, content_loss))

            # 3) total variation loss
            tv_loss = losses.compute_total_variation_loss_l1(outputs)
            self.tv_loss[selected_layer] = tv_loss
            self.total_loss += self.tv_weight * tv_loss
            summaries.add(tf.summary.scalar(
                'tv_loss/decoder_%d' % i, tv_loss))

            image_tiles = tf.concat([inputs, outputs], axis=2)
            image_tiles = preprocessing.batch_mean_image_summation(image_tiles)
            image_tiles = tf.cast(tf.clip_by_value(image_tiles, 0.0, 255.0), tf.uint8)
            summaries.add(tf.summary.image(
                'image_comparison/decoder_%d' % i, image_tiles, max_outputs=8))

        self.summaries = summaries
        return self.total_loss

    def get_training_operations(self, optimizer, global_step,
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
            grads_and_vars, global_step=global_step)
        train_ops.append(train_op)

        self.summaries |= variables_summaries
        self.train_op = tf.group(*train_ops)
        return self.train_op
