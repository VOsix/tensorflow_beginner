import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./data", one_hot=True)
print(mnist.train.num_examples)
tf.logging.set_verbosity(old_v)

FLAGS = tf.app.flags.FLAGS
# tf.app.flags.DEFINE_string("param_name", "default_val", "description")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "")
tf.app.flags.DEFINE_integer("training_epoch", 2, "")
tf.app.flags.DEFINE_integer("batch_size", 100, "")
tf.app.flags.DEFINE_integer("display_step", 1, "")
tf.app.flags.DEFINE_integer("n_hidden_1 ", 256, "")
tf.app.flags.DEFINE_integer("n_hidden_2 ", 256, "")
tf.app.flags.DEFINE_integer("n_input ", 784, "")
tf.app.flags.DEFINE_integer("n_classes ", 10, "")


def main():
    learning_rate = FLAGS.learning_rate
    training_epoch = FLAGS.training_epoch
    batch_size = FLAGS.batch_size
    display_step = FLAGS.display_step
    n_hidden_1 = FLAGS.n_hidden_1
    n_hidden_2 = FLAGS.n_hidden_2
    n_input = FLAGS.n_input
    n_classes = FLAGS.n_classes

    x = tf.placeholder(dtype="float32", shape=[None, n_input])
    y = tf.placeholder(dtype="float32", shape=[None, n_classes])

    weights = {
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
    }

    biases = {

    }




