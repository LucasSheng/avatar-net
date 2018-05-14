from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

from datasets import dataset_utils

import tensorflow as tf

_NUM_SHARDS = 5
_RANDOM_SEED = 0

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'output_dataset_dir', None,
    'The directory where the outputs TFRecords and temporary files are saved')

tf.app.flags.DEFINE_string(
    'input_dataset_dir', None,
    'The directory where the input files are saved.')


def _get_filenames(dataset_dir):
    split_dirs = ['train2014', 'val2014', 'test2014']

    # get the full path to each image
    train_dir = os.path.join(dataset_dir, split_dirs[0])
    validation_dir = os.path.join(dataset_dir, split_dirs[1])
    test_dir = os.path.join(dataset_dir, split_dirs[2])

    train_image_filenames = []
    for filename in os.listdir(train_dir):
        file_path = os.path.join(train_dir, filename)
        train_image_filenames.append(file_path)

    validation_image_filenames = []
    for filename in os.listdir(validation_dir):
        file_path = os.path.join(validation_dir, filename)
        validation_image_filenames.append(file_path)

    test_image_filenames = []
    for filename in os.listdir(test_dir):
        file_path = os.path.join(test_dir, filename)
        test_image_filenames.append(file_path)

    print('Statistics in MSCOCO dataset...')
    print('There are %d images in train dataset' % len(train_image_filenames))
    print('There are %d images in validation dataset' % len(validation_image_filenames))
    print('There are %d images in test dataset' % len(test_image_filenames))

    return train_image_filenames, validation_image_filenames, test_image_filenames


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'MSCOCO_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, image_filenames, dataset_dir):
    assert split_name in ['train', 'validation', 'test']

    num_per_shard = int(math.ceil(len(image_filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = dataset_utils.ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(image_filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                            i + 1, len(image_filenames), shard_id))
                        sys.stdout.flush()
                        # read the image
                        img_filename = image_filenames[i]
                        img_data = tf.gfile.FastGFile(img_filename, 'r').read()
                        img_shape = image_reader.read_image_dims(sess, img_data)
                        example = dataset_utils.image_to_tfexample(
                            img_data, img_filename[-3:], img_shape, img_filename)
                        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation', 'test']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(input_dataset_dir, output_dataset_dir):
    if not tf.gfile.Exists(output_dataset_dir):
        tf.gfile.MakeDirs(output_dataset_dir)

    if _dataset_exists(output_dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    train_image_filenames, validation_image_filenames, test_image_filenames = \
        _get_filenames(input_dataset_dir)

    # randomize the datasets
    random.seed(_RANDOM_SEED)
    random.shuffle(train_image_filenames)
    random.shuffle(validation_image_filenames)
    random.shuffle(test_image_filenames)

    num_train = len(train_image_filenames)
    num_validation = len(validation_image_filenames)
    num_test = len(test_image_filenames)
    num_samples = num_train + num_validation + num_test

    # store the dataset meta data
    dataset_meta_data = {
        'dataset_name': 'MSCOCO',
        'source_dataset_dir': input_dataset_dir,
        'num_of_samples': num_samples,
        'num_of_train': num_train,
        'num_of_validation': num_validation,
        'num_of_test': num_test,
        'train_image_filenames': train_image_filenames,
        'validation_image_filenames': validation_image_filenames,
        'test_image_filenames': test_image_filenames}
    dataset_utils.write_dataset_meta_data(output_dataset_dir, dataset_meta_data)

    _convert_dataset('train', train_image_filenames, output_dataset_dir)
    _convert_dataset('validation', validation_image_filenames, output_dataset_dir)
    _convert_dataset('test', test_image_filenames, output_dataset_dir)


def main(_):
    if not FLAGS.input_dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    run(FLAGS.input_dataset_dir, FLAGS.output_dataset_dir)


if __name__ == '__main__':
    tf.app.run()
