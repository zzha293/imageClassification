import conversion
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle
import data_helper


data_sets = data_helper.load_data()
classes= data_sets['classes']
train_data = data_sets['images_train']
train_label = data_sets['labels_train']
test_data = data_sets['images_test']
test_label = data_sets['labels_test']

#Reshaping data
train_data = train_data.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)
test_data = test_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)


batch_size = 100
image_width = 32
image_height = 32
channels = 3

#placeholder
images_placeholder = tf.placeholder(tf.float32, [None, image_width, image_height, channels])
labels_placeholder = tf.placeholder(tf.int32, [None])
one_hot = tf.one_hot(labels_placeholder,depth=10)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



#initiate a network
we_conv1 = weight_variable([2, 2, 3, 8])
bs_conv1 = bias_variable([8])
layer_conv1 = tf.nn.relu(conv2d(images_placeholder, we_conv1) + bs_conv1)
layer_pool1 = max_pool_2x2(layer_conv1)

we_conv2 = weight_variable([5, 5, 8, 8])
bs_conv2 = bias_variable([8])
layer_conv2 = tf.nn.relu(conv2d(layer_pool1, we_conv2) + bs_conv2)
layer_pool2 = max_pool_2x2(layer_conv2)

we_conv3 = weight_variable([5, 5, 8, 16])
bs_conv3 = bias_variable([16])
layer_conv3 = tf.nn.relu(conv2d(layer_pool2, we_conv3) + bs_conv3)

we_fc2 = weight_variable([16, 10])
bs_fc2 = bias_variable([10])

we_fc1 = weight_variable([8 * 8 * 16, 16])
bs_fc1 = bias_variable([16])
layer_pool3_flat = tf.reshape(layer_conv3, [-1, 8*8*16])
layer_fc1 = tf.nn.relu(tf.matmul(layer_pool3_flat, we_fc1) + bs_fc1)

y_conv = tf.nn.softmax(tf.matmul(layer_fc1, we_fc2) + bs_fc2)


loss = -tf.reduce_sum(one_hot*tf.log(tf.clip_by_value(y_conv,1e-1,1e2)))
optimizer = tf.train.MomentumOptimizer(learning_rate = 0.001, momentum = 0.9).minimize(loss)
correct_pred = tf.equal(tf.argmax(one_hot,1), tf.argmax(y_conv,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


epochs = 100
b_per = 0
row = []


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)

    for e in range(epochs):
        print( "epoch", e)
        avg_loss = 0.0
        for j in range(int(train_data.shape[0]/batch_size)):
            subset=range((j*batch_size),((j+1)*batch_size))
            data = train_data[subset,:,:,:]
            label = train_label[subset]
            _,c = sess.run([optimizer,loss], feed_dict={images_placeholder: data, labels_placeholder: label})


            avg_loss = c/data.shape[0]
            print "Epoch %d, Loss: %.3f" % (e + 1, avg_loss)

            b_per = b_per + 1

            if b_per%10==0 :
                train_accuracy = sess.run(accuracy, feed_dict={images_placeholder: data, labels_placeholder: label})
                #print('Step {:d}, training accuracy {:g}'.format(j, train_accuracy))
                break

        test_accuracy = sess.run(accuracy, feed_dict={images_placeholder: test_data, labels_placeholder: test_label })
        print('Test accuracy {:g}'.format(test_accuracy))