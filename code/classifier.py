"""An example of how to use your own dataset to train a classifier that recognizes people.
"""

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
from sklearn.svm import SVC

def main(args):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            np.random.seed(seed=args.seed)
            dataset = palmPrintsNet.get_dataset(args.data_dir)
            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
            paths, labels = palmPrintsNet.get_image_paths_and_labels(dataset)
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
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

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))

                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.7f' % accuracy)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='...', default='CLASSIFY')
    parser.add_argument('--data_dir', type=str,
        help='..',default='data/classifierdata/palmvein_v1/train')
    parser.add_argument('--model', type=str,
        help='...',default='src/models/palmvein_0111/20180112-230053')
    parser.add_argument('--classifier_filename',
        help='.',default='src/models/new_palmveinclassifer.pkl')
    parser.add_argument('--test_data_dir', type=str,
        help='..',default='data/classifierdata/palmvein_v1/test')
    parser.add_argument('--batch_size', type=int,
        help='...', default=1)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
