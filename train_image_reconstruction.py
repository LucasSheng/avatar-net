from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from models.preprocessing import preprocessing_image
from models import models_factory
from datasets import dataset_utils

slim = tf.contrib.slim


tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')
tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 1,
    'The number of threads used to create the batches.')

# ====================== #
# Training specification #
# ====================== #
tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel',
    'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 100,
    'The frequency with which logs are printed, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the models is saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 120,
    'The frequency with which summaries are saved, in seconds.')
tf.app.flags.DEFINE_integer(
    'batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_integer(
    'max_number_of_steps', None, 'The maximum number of training steps.')

# ============= #
# Dataset Flags #
# ============= #
tf.app.flags.DEFINE_string(
    'dataset_dir', None,
    'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'dataset_name', None,
    'The name of the dataset to load.')
tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train',
    'The name of the train/test split.')

#######################
# Model specification #
#######################
tf.app.flags.DEFINE_string(
    'model_config', None,
    'Directory where the configuration of the models is stored.')

######################
# Optimization Flags #
######################
tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')
tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95, 'The decay rate for adadelta.')
tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')
tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float(
    'opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float(
    'ftrl_learning_rate_power', -0.5, 'The learning rate power.')
tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################
tf.app.flags.DEFINE_string(
    'learning_rate_decay_type', 'exponential',
    'Specififies how the learning rate is decayed. One of "fixed",'
    '"exponential", or "polynomial".')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')
tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'If left as None, the moving averages are not used.')

# ============================ #
# Fine-Tuning Flags
# ============================ #
tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')
tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')
tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS


def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
        num_samples_per_epoch: The number of samples in each epoch of training
        global_step: The global_step tensor.

    Returns:
        A `Tensor` representing the learning rate

    Raises:
        ValueError
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.learning_rate_decay_factor,
            staircase=True,
            name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(
            FLAGS.learning_rate,
            global_step,
            decay_steps,
            FLAGS.end_learning_rate,
            power=1.0,
            cycle=False,
            name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or 'Tensor' learning rate

    Returns:
        An instance of an optimizer

    Raises:
        ValueError: if FLAGS.optimizer is not recognized
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate, rho=FLAGS.adadelta_rho, epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftr1':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def _get_variables_to_train(options):
    """Returns a list of variables to train.

    Args:
        A list of variables to train by the optimizer.
    """
    if options.get('trainable_scopes') is None:
        return tf.trainable_variables()
    else:
        scopes = [scope.strip() for scope in options.get('trainable_scopes').split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def _get_init_fn(options):
    """Returns a function to warm-start the training.

    Note that the init_fn is only run when initializing the models during the
    very first global step.

    Returns:
        An init function
    """
    if options.get('checkpoint_path') is None:
        return None

    # Warn the user if a checkpoint exists in the train_dir. Then we'll be
    # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists '
            'in %s' % FLAGS.train_dir)
        return None

    exclusions = []
    if options.get('checkpoint_exclude_scopes'):
        # remove space and comma
        exclusions = [scope.strip()
                      for scope in options.get('checkpoint_exclude_scopes').split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(options.get('checkpoint_path')):
        checkpoint_path = tf.train.latest_checkpoint(options.get('checkpoint_path'))
    else:
        checkpoint_path = options.get('checkpoint_path')

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=options.get('ignore_missing_vars'))


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with'
                         ' --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        global_step = slim.create_global_step()  # create the global step

        ######################
        # select the dataset #
        ######################
        dataset = dataset_utils.get_split(
            FLAGS.dataset_name,
            FLAGS.dataset_split_name,
            FLAGS.dataset_dir)

        ######################
        # create the network #
        ######################
        # parse the options from a yaml file
        model, options = models_factory.get_model(FLAGS.model_config)

        ####################################################
        # create a dataset provider that loads the dataset #
        ####################################################
        # dataset provider
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20*FLAGS.batch_size,
            common_queue_min=10*FLAGS.batch_size)
        [image] = provider.get(['image'])
        image_clip = preprocessing_image(
            image,
            model.training_image_size,
            model.training_image_size,
            model.content_size,
            is_training=True)
        image_clip_batch = tf.train.batch(
            [image_clip],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5*FLAGS.batch_size)

        # feque queue the inputs
        batch_queue = slim.prefetch_queue.prefetch_queue([image_clip_batch])

        ###########################################
        # build the models based on the given data #
        ###########################################
        images = batch_queue.dequeue()
        total_loss = model.build_train_graph(images)

        ####################################################
        # gather the operations for training and summaries #
        ####################################################
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # configurate the moving averages
        if FLAGS.moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        # gather the optimizer operations
        learning_rate = _configure_learning_rate(
            dataset.num_samples, global_step)
        optimizer = _configure_optimizer(learning_rate)
        summaries.add(tf.summary.scalar('learning_rate', learning_rate))

        if FLAGS.moving_average_decay:
            update_ops.append(variable_averages.apply(moving_average_variables))

        # training operations
        train_op = model.get_training_operations(
            optimizer, global_step, _get_variables_to_train(options))
        update_ops.append(train_op)

        # gather the training summaries
        summaries |= set(model.summaries)

        # gather the update operation
        update_op = tf.group(*update_ops)
        watched_loss = control_flow_ops.with_dependencies(
            [update_op], total_loss, name='train_op')

        # merge the summaries
        summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        ##############################
        # start the training process #
        ##############################
        slim.learning.train(
            watched_loss,
            logdir=FLAGS.train_dir,
            init_fn=_get_init_fn(options),
            summary_op=summary_op,
            number_of_steps=FLAGS.max_number_of_steps,
            log_every_n_steps=FLAGS.log_every_n_steps,
            save_summaries_secs=FLAGS.save_summaries_secs,
            save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
    tf.app.run()
