# Import MNIST
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
mnist = input_data.read_data_sets("./data", one_hot=True)

# Load data
X_train = mnist.train.images
Y_train = mnist.train.labels
X_test = mnist.test.images
Y_test = mnist.test.labels
batch_X, batch_Y = mnist.train.next_batch(64)
print(batch_X.shape, batch_Y.shape)

# 可视化
