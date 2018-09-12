from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import data_helper
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt



data_sets = data_helper.load_data()
classes= data_sets['classes']
train_data = data_sets['images_train']
train_label = data_sets['labels_train']
test_data = data_sets['images_test']
test_label = data_sets['labels_test']

data_sets2 = data_helper.reshape_data(data_sets)
X_train2 = data_sets2['images_train']
y_train2 = data_sets2['labels_train']
X_test2 = data_sets2['images_test']
y_test2 = data_sets2['labels_test']

hidden1 = 120
learning_rate =0.001
max_steps = 1200

reg_constant = 0.1
batch_size= 400

IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()


num_classes = len(classes)
num_images = 5

def plot_images():
    for y, cla in enumerate(classes):
        idxs = np.flatnonzero(train_label == y)
        idxs = np.random.choice(idxs, num_images, replace = False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(num_images, num_classes, plt_idx)
            plt.imshow(X_train2[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cla)
    plt.show()

#show the graph
plot_images()


def inference(images, image_pixels, hidden_units, classes, reg_constant=0):
  # Layer 1
  with tf.variable_scope('Layer1'):
    # Define the variables
    weights = tf.get_variable(name='weights',shape=[image_pixels, hidden_units],
      initializer=tf.truncated_normal_initializer(stddev=1.0 / np.sqrt(float(image_pixels))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

    biases = tf.Variable(tf.zeros([hidden_units]), name='biases')
    hidden = tf.nn.relu(tf.matmul(images, weights) + biases)

  # Layer 2, Hidden
  with tf.variable_scope('Layer2'):
    # Define variables
    weights = tf.get_variable('weights', [hidden_units, classes],
      initializer=tf.truncated_normal_initializer(
        stddev=1.0 / np.sqrt(float(hidden_units))),
      regularizer=tf.contrib.layers.l2_regularizer(reg_constant))

    biases = tf.Variable(tf.zeros([classes]), name='biases')
    logits = tf.matmul(hidden, weights) + biases

    tf.summary.histogram('logits', logits)
  return logits


def loss(logits, labels):
  with tf.name_scope('Loss'):
    cross_entropy = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy'))

    loss = cross_entropy + tf.add_n(tf.get_collection(
      tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.scalar('loss', loss)

  return loss


def training(loss, learning_rate):
  global_step = tf.Variable(0, name='global_step', trainable=False)

  train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    loss, global_step=global_step)

  return train_step


def evaluation(logits, labels):

  with tf.name_scope('Accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits,1), labels)
    accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('train_accuracy', accuracy)
  return accuracy


images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],name='images')
labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')
logits = inference(images_placeholder, IMAGE_PIXELS,hidden1, CLASSES, reg_constant=reg_constant)


accuracy = evaluation(logits, labels_placeholder)
loss = loss(logits, labels_placeholder)
train_step = training(loss, learning_rate)
summary = tf.summary.merge_all()
saver = tf.train.Saver()

def main():
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    zipped_data = zip(train_data, train_label)
    batches = data_helper.gen_batch(list(zipped_data), batch_size, max_steps)

    for i in range(max_steps):

      batch = next(batches)
      images_batch, labels_batch = zip(*batch)
      feed_dict = {
        images_placeholder: images_batch,
        labels_placeholder: labels_batch
      }

      # Periodically print out the model's current accuracy
      if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict=feed_dict)
        print('Step {:d}, training accuracy {:g}'.format(i, train_accuracy))

      sess.run([train_step, loss], feed_dict=feed_dict)


    test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: test_data, labels_placeholder: test_label})
    print('Test accuracy {:g}'.format(test_accuracy))


if __name__ == '__main__':
  main()