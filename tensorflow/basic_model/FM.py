# -*- coding: utf-8 -*-
# Author: lzjiang
import json
import tensorflow as tf
import logging
import sys
import os
import pdb
import pprint

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
FLAGS = flags.FLAGS

class FM(object):
    def __init__(self):

