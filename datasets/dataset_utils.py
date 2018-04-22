from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import tensorflow as tf

slim = tf.contrib.slim

_META_DATA_FILENAME = 'dataset_meta_data.txt'

_FILE_PATTERN = '%s_%s_*.tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'shape': 'The shape of the image.'
}


def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def image_to_tfexample(image_data, image_format, image_shape, image_filename):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/shape': int64_feature(image_shape),
        'image/filename': bytes_feature(image_filename),
    }))


def write_dataset_meta_data(dataset_dir, dataset_meta_data,
                            filename=_META_DATA_FILENAME):
    meta_filename = os.path.join(dataset_dir, filename)
    with open(meta_filename, 'wb') as f:
        yaml.dump(dataset_meta_data, f)
        print('Finish writing the dataset meta data.')


def has_dataset_meta_data_file(dataset_dir, filename=_META_DATA_FILENAME):
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_dataset_meta_data(dataset_dir, filename=_META_DATA_FILENAME):
    meta_filename = os.path.join(dataset_dir, filename)
    with open(meta_filename, 'rb') as f:
        dataset_meta_data = yaml.load(f)
        print('Finish loading the dataset meta data of [%s].' %
              dataset_meta_data.get('dataset_name'))
        return dataset_meta_data


def get_split(dataset_name,
              split_name,
              dataset_dir,
              file_pattern=None,
              reader=None):
    if split_name not in ['train', 'validation']:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
        file_pattern = os.path.join(dataset_dir, file_pattern % (
            dataset_name, split_name))

    # read the dataset meta data
    if has_dataset_meta_data_file(dataset_dir):
        dataset_meta_data = read_dataset_meta_data(dataset_dir)
        num_samples = dataset_meta_data.get('num_of_' + split_name)
    else:
        raise ValueError('No dataset_meta_data file available in %s' % dataset_dir)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/shape': tf.FixedLenFeature((3,), tf.int64, default_value=(224, 224, 3)),
        'image/filename': tf.FixedLenFeature([], tf.string, default_value=''),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(
            'image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'filename': slim.tfexample_decoder.Tensor('image/filename')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)


class ImageReader(object):
    """helper class that provides tensorflow image coding utilities."""
    def __init__(self):
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decode_image = tf.image.decode_image(self._decode_data, channels=0)

    def read_image_dims(self, sess, image_data):
        image = self.decode_image(sess, image_data)
        return image.shape

    def decode_image(self, sess, image_data):
        image = sess.run(self._decode_image,
                         feed_dict={self._decode_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


class ImageCoder(object):
    """helper class that provides Tensorflow Image coding utilities,
    also works for corrupted data with incorrected extension
    """
    def __init__(self):
        self._decode_data = tf.placeholder(dtype=tf.string)
        self._decode_image = tf.image.decode_image(self._decode_data, channels=0)
        self._encode_jpeg = tf.image.encode_jpeg(self._decode_image, format='rgb', quality=100)

    def decode_image(self, sess, image_data):
        # verify the image from the image_data
        status = False
        try:
            # decode image and verify the data
            image = sess.run(self._decode_image,
                             feed_dict={self._decode_data: image_data})
            image_shape = image.shape
            assert len(image_shape) == 3
            assert image_shape[2] == 3
            # encode as RGB JPEG image string and return
            image_string = sess.run(self._encode_jpeg, feed_dict={self._decode_data: image_data})
            status = True
        except BaseException:
            image_shape, image_string = None, None
        return status, image_string, image_shape
