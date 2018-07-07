import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

Xtr, Ytr = mnist.train.next_batch(5000)  # training
Xte, Yte = mnist.test.next_batch(400)  # testing

# tf.placeholder(dtype, shape=None, name=None)
# dtype,常用类型tf.float32,tf.float64
# shape,默认为None,[None,784]表示列为784,行不定
xtr = tf.placeholder("float", [None, 784])  # None代表将可选多个样本,[60,784]表示选取60个样本，每个样本都是784列
xte = tf.placeholder("float", [784])  # test只选用一样本

# tf基本算数运算
# 加法tf.add(x,y,name=None)
# 减法tf.subtract(x,y,name=None)
# 乘法tf.multiply(x,y,name=None)
# 除法tf.div(x,y,name=None)
# 取模tf.mod(x,y,name=None)
# 求绝对值tf.abs(x,name=None)
# 取负tf.negative(x,name=None)
# 返回符号tf.sign(x,name=None):x>0=>1;x<0=>-1;0=>0
# 取倒数tf.reciprocal(x,name=None)
# 平方tf.square(x,name=None)
# 舍入最接近的整数tf.round(x,name=None)
# 幂次方tf.pow(x,y,name=None)
#distince = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
distince = tf.reduce_sum(tf.square(tf.add(xtr, tf.negative(xte))), reduction_indices=1)

# 求距离最短的训练样本的索引
prediction = tf.arg_min(distince, 0)
accuracy = 0
# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for i in range(len(Xte)):
        nn_index = sess.run(prediction, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), "True Class:", np.argmax(Yte[i]))
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1. / len(Xte)
    print("Finish!!")
    print("准确率:", accuracy)
