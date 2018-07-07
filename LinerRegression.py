import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# random num
rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 3000
display_step = 100

# Train data
train_X = np.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                      7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = np.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                      2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
# n_samples_num
n_samples = train_X.shape[0]
print(train_X[0:1])
print(train_Y[0:1])

X = tf.placeholder(dtype="float32", shape=[None])
Y = tf.placeholder(dtype="float32", shape=[None])

W = tf.Variable(0.5, name="weight")
b = tf.Variable(0.5, name="bias")

pred = tf.add(tf.multiply(W, X), b)

loss1 = tf.reduce_sum(tf.square(pred - Y) / n_samples)/2
loss2 = tf.reduce_mean(tf.square(pred - Y))/2
tf.summary.scalar("loss", loss1)
merged_summary = tf.summary.merge_all()

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log", sess.graph)
    sess.run(init)
    for i in range(training_epochs):
        _, summary = sess.run([train_step, merged_summary], feed_dict={X: train_X, Y: train_Y})
        writer.add_summary(summary, i)

        if (i + 1) % display_step == 0:
            c1, c2 = sess.run([loss1, loss2], feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (i + 1), "loss1= {:.9f} loss2= {:.9f}".format(c1, c2), "W=", sess.run(W), "b=",
                  sess.run(b))
    print("Optimization Finished!")

    train_loss = sess.run(loss1, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", train_loss, "W=", sess.run(W), "b=", sess.run(b), '\n')

    plt.plot(train_X, train_Y, 'ro', label="original data")
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label="Fitted Line")
    plt.legend()
    plt.show()

    test_X = np.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    test_Y = np.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])
    print("Testing... (Mean square loss Comparison)")
    testing_loss = sess.run(tf.reduce_mean(tf.square(pred - Y)) / 2,
                            feed_dict={X: test_X, Y: test_Y})
    print("Testing cost=", testing_loss)
    print("Absolute mean square loss difference:", abs(train_loss - testing_loss))
    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')

    plt.legend()

    plt.show()
