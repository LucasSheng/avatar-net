from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    cropped_shape = control_flow_ops.with_dependencies(
        [rank_assertion],
        tf.stack([crop_height, crop_width, original_shape[2]]))

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    image = control_flow_ops.with_dependencies(
        [size_assertion],
        tf.slice(image, offsets, cropped_shape))
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    image_shape = control_flow_ops.with_dependencies(
        [rank_assertions[0]],
        tf.shape(image_list[0]))
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        shape = control_flow_ops.with_dependencies([rank_assertions[i]],
                                                   tf.shape(image))
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    max_offset_height = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_height - crop_height + 1, []))
    max_offset_width = control_flow_ops.with_dependencies(
        asserts, tf.reshape(image_width - crop_width + 1, []))
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


def _central_crop(image_list, crop_height, crop_width):
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def _mean_image_subtraction(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def preprocessing_for_train(image, output_height, output_width, resize_side):
    image = _aspect_preserving_resize(image, resize_side)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocessing_for_eval(image, output_height, output_width, resize_side):
    image = _aspect_preserving_resize(image, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])


def preprocessing_image(image, output_height, output_width,
                        resize_side=_RESIZE_SIDE_MIN, is_training=False):
    if is_training:
        return preprocessing_for_train(image, output_height, output_width, resize_side)
    else:
        return preprocessing_for_eval(image, output_height, output_width, resize_side)


#########################
# personal modification #
#########################
def mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """works for one single image with dynamic shapes"""
    num_channels = 3
    channels = tf.split(images, num_channels, axis=2)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=2)


def mean_image_summation(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """works for one single image with dynamic shapes"""
    num_channels = 3
    channels = tf.split(image, num_channels, axis=2)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, axis=2)


def batch_mean_image_subtraction(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_channels, axis=3)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(channels, axis=3)


def batch_mean_image_summation(images, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    if images.get_shape().ndims != 4:
        raise ValueError('Input must be of size [batch, height, width, C>0')
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(images, num_channels, axis=3)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(channels, axis=3)


def image_normalization(images, means=(_R_MEAN, _G_MEAN, _B_MEAN), scale=127.5):
    """rescale the images so that their magnitude ranging from [-1, 1]"""
    if images.get_shape().ndims == 4:
        return tf.div(batch_mean_image_subtraction(images, means), scale)
    elif images.get_shape().ndims == 3:
        return tf.div(mean_image_subtraction(images, means), scale)
    else:
        raise ValueError('Input must be of dimensions 3 or 4')


def aspect_preserving_resize(image, smallest_side):
    return _aspect_preserving_resize(image, smallest_side)
