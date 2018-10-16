"""
Functions for training palmvein cnn

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import palmPrintsNet
import palmPrints_val
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops

def main(args):
    #Importing network based on model.def path
    network = importlib.import_module(args.model_def)
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)

    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    #Create a model file directory
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    palmPrintsNet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
    src_path,_ = os.path.split(os.path.realpath(__file__))
    palmPrintsNet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = palmPrintsNet.get_dataset(args.data_dir)
    #Total class's number
    nrof_classes = len(train_set)
    print('model_path: %s' % model_dir)
    print('log_path: %s' % log_dir)

    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        image_list, label_list = palmPrintsNet.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The dataset should not be empty'
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]

        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,shuffle=True, seed=None, capacity=32)

        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')

        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)

        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')

        nrof_preprocess_threads = 4
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=1)
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])
        image_batch, label_batch = tf.train.batch_join(images_and_labels, batch_size=batch_size_placeholder,shapes=[(160, 160, 1), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,allow_smaller_final_batch=True)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        print('the number of class: %d' % nrof_classes)
        print('the number of image: %d' % len(image_list))

        #perlogits is a 128-v feature vector
        prelogits, _ = network.inference(image_batch, args.keep_probability,phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size,weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                weights_regularizer=slim.l2_regularizer(args.weight_decay),scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        #center loss
        if args.center_loss_factor>0.0:
            prelogits_center_loss, _ = palmPrintsNet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)
        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)
        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        #total loss
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        #train
        train_op = palmPrintsNet.train(total_loss, global_step, args.optimizer,
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)
        with sess.as_default():
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size

                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)

                save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)
    return model_dir

def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step,
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0

    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = palmPrintsNet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)

    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run([loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_base_dir', type=str,
        help='Directory where to write event logs.', default='logs/palmvein_0111/')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='src/models/palmvein_0111/')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',default='data/deeplearningdata/palmvain_v1/train')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=100)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=24)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=300)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)

    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=0.8)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=5e-4)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.9)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='RMSPROP')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=-1.0)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms',
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule_classifier_casia.txt')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
