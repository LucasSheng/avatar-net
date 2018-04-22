from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import scipy.misc
import numpy as np
import tensorflow as tf

from models import models_factory
from models import preprocessing

from PIL import Image

slim = tf.contrib.slim


tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'tmp/tfmodel',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')
tf.app.flags.DEFINE_string(
    'eval_dir', 'tmp/tfmodel',
    'Directory where the results are saved to.')
tf.app.flags.DEFINE_string(
    'content_dataset_dir', None,
    'The content directory where the test images are stored.')
tf.app.flags.DEFINE_string(
    'style_dataset_dir', None,
    'The style directory where the style images are stored.')

# choose the model configuration file
tf.app.flags.DEFINE_string(
    'model_config_path', None,
    'The path of the model configuration file.')
tf.app.flags.DEFINE_float(
    'inter_weight', 1.0,
    'The blending weight of the style patterns in the stylized image')

FLAGS = tf.app.flags.FLAGS


def get_image_filenames(dataset_dir):
    """helper fn that provides the full image filenames from the dataset_dir"""
    image_filenames = []
    for filename in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, filename)
        image_filenames.append(file_path)
    return image_filenames


def image_reader(filename):
    """help fn that provides numpy image coding utilities"""
    img = scipy.misc.imread(filename).astype(np.float)
    if len(img.shape) == 2:
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def imsave(filename, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(filename, quality=95)


def main(_):
    if not FLAGS.content_dataset_dir:
        raise ValueError('You must supply the content dataset directory '
                         'with --content_dataset_dir')
    if not FLAGS.style_dataset_dir:
        raise ValueError('You must supply the style dataset directory '
                         'with --style_dataset_dir')

    if not FLAGS.checkpoint_dir:
        raise ValueError('You must supply the checkpoints directory with '
                         '--checkpoint_dir')

    if tf.gfile.IsDirectory(FLAGS.checkpoint_dir):
        checkpoint_dir = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    else:
        checkpoint_dir = FLAGS.checkpoint_dir

    if not tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.MakeDirs(FLAGS.eval_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # define the model
        style_model, options = models_factory.get_model(FLAGS.model_config_path)

        # predict the stylized image
        inp_content_image = tf.placeholder(tf.float32, shape=(None, None, 3))
        inp_style_image = tf.placeholder(tf.float32, shape=(None, None, 3))

        # preprocess the content and style images
        content_image = preprocessing.aspect_preserving_resize(
            inp_content_image,
            smallest_side=style_model.content_size)
        content_image = preprocessing.mean_image_subtraction(content_image)
        content_image = tf.expand_dims(content_image, axis=0)
        style_image = preprocessing.preprocessing_image(
            inp_style_image,
            448,
            448,
            style_model.style_size)
        style_image = tf.expand_dims(style_image, axis=0)

        # style transfer
        stylized_image = style_model.transfer_styles(content_image, style_image, inter_weight=FLAGS.inter_weight)
        stylized_image = tf.squeeze(stylized_image, axis=0)

        # gather the test image filenames and style image filenames
        style_image_filenames = get_image_filenames(FLAGS.style_dataset_dir)
        content_image_filenames = get_image_filenames(FLAGS.content_dataset_dir)

        # starting inference of the images
        init_fn = slim.assign_from_checkpoint_fn(
          checkpoint_dir, slim.get_model_variables(), ignore_missing_vars=True)
        with tf.Session() as sess:
            # initialize the graph
            init_fn(sess)

            nn = 0.0
            total_time = 0.0
            # style transfer for each image based on one style image
            for i in range(len(style_image_filenames)):
                # gather the storage folder for the style transfer
                style_label = style_image_filenames[i].split('/')[-1]
                style_label = style_label.split('.')[0]
                style_dir = os.path.join(FLAGS.eval_dir, style_label)

                if not tf.gfile.Exists(style_dir):
                    tf.gfile.MakeDirs(style_dir)

                # get the style image
                np_style_image = image_reader(style_image_filenames[i])
                print('Starting transferring the style of [%s]' % style_label)

                for j in range(len(content_image_filenames)):
                    # gather the content image
                    np_content_image = image_reader(content_image_filenames[j])

                    start_time = time.time()
                    np_stylized_image = sess.run(stylized_image,
                                                 feed_dict={inp_content_image: np_content_image,
                                                            inp_style_image: np_style_image})
                    incre_time = time.time() - start_time
                    nn += 1.0
                    total_time += incre_time
                    print("---%s seconds ---" % (total_time/nn))

                    output_filename = os.path.join(
                        style_dir, content_image_filenames[j].split('/')[-1])
                    imsave(output_filename, np_stylized_image)
                    print('Style [%s]: Finish transfer the image [%s]' % (
                        style_label, content_image_filenames[j]))


if __name__ == '__main__':
    tf.app.run()
