#测试
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import palmPrintsNet
import os
import sys
import math
import pickle

def main(args):
    with tf.Graph().as_default():
        with tf.device("/cpu:0"):
            with tf.Session() as sess:
                np.random.seed(seed=args.seed)
                dataset = palmPrintsNet.get_dataset(args.data_dir)
                paths, labels = palmPrintsNet.get_image_paths_and_labels(dataset)
                print('Number of classes: %d' % len(dataset))
                print('Number of images: %d' % len(paths))
                print('Loading feature extraction model')
                palmPrintsNet.load_model(args.model)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                embedding_size = embeddings.get_shape()[1]

                # Run forward pass to calculate embeddings
                print('Calculating features for images')
                nrof_images = len(paths)
                emb_array = np.zeros((nrof_images, embedding_size))
                for i in range(nrof_images):
                    print(i)
                    start_index = i
                    end_index = min((i+1), nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = palmPrintsNet.load_data(paths_batch, False, False, args.image_size)
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

                np.savetxt("palmvein_test.txt", emb_array);

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
        help='...', default='data/deeplearningdata/palmvein_v1/test')
    parser.add_argument('--model', type=str,
        help='...', default='src/models/palmvein_0111/20180112-230053')
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--batch_size', type=int,
        help='...', default=1)
    parser.add_argument('--image_size', type=int,
        help='...', default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
