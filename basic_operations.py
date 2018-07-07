import tensorflow as tf

# constant
a = tf.constant(2)
b = tf.constant(3)
with tf.Session() as sess:
    print("a: %i" % sess.run(a), "b:%i" % sess.run(b))
    print("Addition with constants: %i" % sess.run(a + b))
    print("Multiplication with constants: %i" % sess.run(a * b))

# placeholder
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
add = tf.add(a, b)
mul = tf.multiply(a, b)

with tf.Session() as sess:
    print("a:%i" % sess.run(a, feed_dict={a: 2}))
    print("b:%i" % sess.run(b, feed_dict={b: 3}))
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

# matrix
matrix1 = tf.constant([[3., 3.]])#1*2
matrix2 = tf.constant([[2.0], [2.0]])#2*1
product = tf.matmul(matrix1,matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

a =tf.constant([0.9, 2.5, 2.3, 1.5, -4.5])
with tf.Session() as sess:
    print(sess.run(tf.round(a)))