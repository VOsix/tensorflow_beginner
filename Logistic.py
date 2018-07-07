# coding:utf-8

' Logistic '
__author__ = 'lzjiang'



import logging
import numpy as np
import tensorflow as tf
import pdb

# tensorflow中日志控制set_verbosity与get_verbosity配合
tf.logging.set_verbosity(tf.logging.ERROR)

# #配置logging基本的设置，然后在控制台输出日志
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s [%(filename)sline:%(lineno)d][%(levelname)s] %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S')


# #将日志写入到文件
# logger = logging.getLogger(__name__)
# logger.setLevel(level = logging.INFO)
# handler = logging.FileHandler("./log.txt")
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s [%(filename)sline:%(lineno)d][%(levelname)s] %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
#
# #将日志写入到控制台
# console = logging.StreamHandler()
# console.setLevel(logging.ERROR)
# logger.addHandler(console)

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "")
tf.app.flags.DEFINE_integer("training_epoch", 2, "")
tf.app.flags.DEFINE_integer("batch_size", 100, "")
tf.app.flags.DEFINE_integer("display_step", 1, "")

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)
print(mnist.train.num_examples)

# learning_rate = 0.01
# training_epoch = 100
# batch_size = 1
# display_step = 1

# pdb.set_trace()
# a1 = np.array([[1., 10.], [10., 100.]], dtype=np.float32)

x = tf.placeholder(dtype="float32", shape=[None, 784])
y = tf.placeholder(dtype="float32", shape=[None, 10])

W = tf.Variable(tf.random_uniform([784, 10]))
b = tf.Variable(tf.random_uniform([10]))


def main():
    learning_rate = FLAGS.learning_rate
    training_epoch = FLAGS.training_epoch
    batch_size = FLAGS.batch_size
    display_step = FLAGS.display_step

    print("learning_rate is ", learning_rate)
    print("training_epoch", training_epoch)
    print("training_epoch", training_epoch)
    print("display_step", display_step)

    prediction = tf.nn.softmax(tf.matmul(x, W) + b)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("./log/Logistic", sess.graph)
        sess.run(init)
        for epoch in range(training_epoch):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += c / total_batch

            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost", "{:.9f}".format(avg_cost))

            summary = tf.Summary(value=[tf.Summary.Value(tag="avg_cost", simple_value=avg_cost)])
            writer.add_summary(summary, epoch)

        print("Optimization Finished!")

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))


def test():
    print("Hello!")


if __name__ == '__main__':
    main()
    test()
