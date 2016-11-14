from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import numpy as np

#from tensorflow.examples.tutorials.mnist import input_data
import NeuralNetwork.TensorFlow.input_data as input_data
import NeuralNetwork.TensorFlow.mnist as mst

# Arguments Definition

learning_rate = 0.01
max_steps = 2000
#hidden1 = 300
batch_size = 100
fake_data = False
train_dir = 'data'

# Load data_set
data_sets = input_data.read_data_sets(train_dir, fake_data)

def getTrainedWeight(hidden1):
    with tf.Graph().as_default() as g:
        # Generate placeholders for the images and labels.
        images_placeholder, labels_placeholder = placeholder_inputs(batch_size)

        # Build a Graph that computes predictions from the inference model.
        logits = mst.inference2(images_placeholder,hidden1)
        # Add to the Graph the Ops for loss calculation.
        loss = mst.loss(logits, labels_placeholder)

        # Add to the Graph the Ops that calculate and apply gradients.
        train_op = mst.training(loss, learning_rate)

        # Add the Op to compare the logits to the labels during evaluation.
        eval_correct = mst.evaluation(logits, labels_placeholder)
        init = tf.initialize_all_variables()

        # Create a saver for writing training checkpoints.
        # saver = tf.train.Saver()

        # Create a session for running Ops on the Graph.
        sess = tf.Session()
        sess.run(init)
        
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train, batch_size, images_placeholder, labels_placeholder)
            _, loss_value, logit_val = sess.run([train_op, loss, logits], feed_dict=feed_dict)
            
    #         duration = time.time() - start_time       
            # Write the summaries and print an overview fairly often.
    #         if step % 200 == 0:
    #         # Print status to stdout.
    #             print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))

            # Save a checkpoint and evaluate the model periodically.
    #         if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
    #             print('Print logits:')
    #             print('Training Data Eval:')
    #             do_eval(sess,
    #                     eval_correct,
    #                     images_placeholder,
    #                     labels_placeholder,
    #                     data_sets.train)
            
    #             # Evaluate against the validation set.
    #             print('Validation Data Eval:')
    #             do_eval(sess,
    #                     eval_correct,
    #                     images_placeholder,
    #                     labels_placeholder,
    #                     data_sets.validation)
    #             # Evaluate against the test set.
    #             print('Test Data Eval:')
    #             do_eval(sess,
    #                     eval_correct,
    #                     images_placeholder,
    #                     labels_placeholder,
    #                     data_sets.test)
            
    print('Step %d: loss = %.2f' % (step, loss_value))    
    
    w1 = g.get_collection(tf.GraphKeys.VARIABLES, "hidden1/weights")
    w2 = g.get_collection(tf.GraphKeys.VARIABLES, "softmax_linear/weights")
    w1_val,w2_val = sess.run([w1,w2])

    return w1_val[0],w2_val[0],loss_value,logit_val



                 
def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mst.IMAGE_PIXELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, batch_size, images_pl, labels_pl):
  images_feed, labels_feed = data_set.next_batch(batch_size,fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  print(steps_per_epoch, num_examples)
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set, batch_size,
                               images_placeholder,
                               labels_placeholder)
    count = sess.run(eval_correct, feed_dict=feed_dict)
    true_count += count 
  precision = true_count / num_examples
  print(true_count / num_examples)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
