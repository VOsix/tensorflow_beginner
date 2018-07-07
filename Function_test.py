import numpy as np
import tensorflow as tf

a = np.random.random((2, 3))
b = np.random.random((1, 3))

a1 = np.array([[1., 2., 3.], [2., 3., 4.]], dtype=np.float32)
b1 = np.array([[1., 1., 1.]], dtype=np.float32)
c1 = np.array([[[[1], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]]]], dtype=np.float32)
d1 = np.array([[[[1], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]]],
               [[[1], [2], [3], [4]],
                [[5], [6], [7], [8]],
                [[9], [10], [11], [12]]]], dtype=np.float32)
e1 = np.array([[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]],
               [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]], dtype=np.float32)
print(a, b)
print(a1, b1)
print(a1.shape, a1.ndim, a1.size)
print(c1.shape, c1.ndim, c1.size)
print(d1.shape, d1.ndim, d1.size)
print(e1.shape, e1.ndim, e1.size)

A0 = tf.placeholder("float", [None, 3])
A1 = tf.placeholder("float", [None, 3, 4])
A2 = tf.placeholder("float", [None, 3, 4, 1])
B = tf.placeholder("float", [1, 3])
with tf.Session() as sess:
    # print(sess.run(tf.add(A, tf.negative(B)), feed_dict={A: a1, B: b1}))
    # print(sess.run(tf.abs(tf.add(A, tf.negative(B))), feed_dict={A: a1, B: b1}))
    # print(sess.run(tf.square(tf.add(A, tf.negative(B))), feed_dict={A: a1, B: b1}))
    # print(sess.run(tf.reduce_sum(tf.abs(tf.add(A, tf.negative(B))), reduction_indices=1), feed_dict={A: a1, B: b1}))  # 按行求和
    # print(sess.run(tf.reduce_sum(tf.abs(tf.add(A, tf.negative(B))), reduction_indices=0), feed_dict={A: a1, B: b1}))  # 按列求和
    # print(sess.run(tf.reduce_sum(tf.square(tf.add(A, tf.negative(B))), reduction_indices=1), feed_dict={A: a1, B: b1}))  # 按行求和

    #reduce_sum
    print(sess.run(tf.reduce_sum(A0, reduction_indices=0), feed_dict={A0: a1}))
    print(sess.run(tf.reduce_sum(A0, reduction_indices=1), feed_dict={A0: a1}))
    #
    # print(sess.run(tf.reduce_sum(A1, reduction_indices=0), feed_dict={A1: e1}))
    # print(sess.run(tf.reduce_sum(A1, reduction_indices=1), feed_dict={A1: e1}))
    # print(sess.run(tf.reduce_sum(A1, reduction_indices=2), feed_dict={A1: e1}))
    #
    # print(sess.run(tf.reduce_sum(A2, reduction_indices=0), feed_dict={A2: d1}))
    # print(sess.run(tf.reduce_sum(A2, reduction_indices=1), feed_dict={A2: d1}))
    # print(sess.run(tf.reduce_sum(A2, reduction_indices=2), feed_dict={A2: d1}))
    # print(sess.run(tf.reduce_sum(A2, reduction_indices=3), feed_dict={A2: d1}))
    print()
