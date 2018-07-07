from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import json
import math
import time
import inspect
from six.moves import urllib
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.estimator.export.export_lib import ServingInputReceiver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(filename)sline:%(lineno)d][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
tf.logging.set_verbosity(tf.logging.INFO)

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("run_mode", "local", "the run mode, one of local and distributed")
flags.DEFINE_string("job_type", "train",
                    "the job type, one of train, eval, train_and_eval, export_savedmodel and predict")

flags.DEFINE_string("train_data", "./train_data", "the train data, delimited by comma")
flags.DEFINE_string("eval_data", "./eval_data", "the eval data, delimited by comma")
flags.DEFINE_integer("batch_size", 2048, "the batch size")
flags.DEFINE_integer("feature_size", 100000, "the feature size")

flags.DEFINE_string("model_type", "wide", "the model type, one of wide, deep and wide_and_deep")
flags.DEFINE_string("model_dir", "./model_dir", "the model dir")
flags.DEFINE_bool("cold_start", False, "True: cold start; False: start from the latest checkpoint")
flags.DEFINE_string("export_savedmodel", "./export_model", "the export savedmodel directory, used for tf serving")
flags.DEFINE_string("savedmodel_mode", "raw", "the savedmodel mode, one of raw and parsing")
flags.DEFINE_integer("eval_ckpt_id", 0, "the checkpoint id in model dir for evaluation, 0 is the latest checkpoint")

# Configurations for a distributed run
flags.DEFINE_string("worker_hosts", "", "the worker hosts, delimited by comma")
flags.DEFINE_string("ps_hosts", "", "the ps hosts, delimited by comma")
flags.DEFINE_string("task_type", "worker", "the task type, one of worker and ps")
flags.DEFINE_integer("task_index", 0, "the task index, starting from 0")


class WideAndDeepModel(object):
    def __init__(self,
                 model_type="wide",
                 model_dir=None,
                 feature_size=100000,
                 run_config=None):
        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        if run_config is None:
            self.run_config = self.build_run_config()
        else:
            self.run_config = run_config

    def build_run_config(self):
        run_config = tf.contrib.learn.RunConfig(
            tf_random_seed=1,  # Random seed for TensorFlow initializers
            save_summary_steps=100,  # Save summaries every this many steps
            save_checkpoints_secs=None,  # Save checkpoints every this many seconds
            save_checkpoints_steps=100,  # Save checkpoints every this many steps
            keep_checkpoint_max=5,  # The maximum number of recent checkpoint files to keep
            keep_checkpoint_every_n_hours=10000,  # Number of hours between each checkpoint to be saved
            log_step_count_steps=100)  # The frequency, in number of global steps
        return run_config

    def build_feature_dict(self):
        feature_dict = {}
        feature_dict["user_adc_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_adc_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_adoff_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_adoff_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_news_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_news_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_tt_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_tt_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_gg_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_gg_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_sea_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_dmp0_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_dmp1_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_dmp1_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_dmp2_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_dmp2_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_inl_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_inl_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["user_cpd_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["date_wh_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["user_titon_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["ad1_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["ad1_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["ad2_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["ad2_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        feature_dict["ad3_features"] = tf.placeholder(dtype=tf.string, shape=(None, self.feature_size))
        feature_dict["ad3_weights"] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))
        return feature_dict

    # 1:click,(2-3:ulabel_adc,4-5:ulabel_adOff,6-7:ulabel_news,8-9:ulabel_tt,10-11:ulabel_gg,12:ulabel_ser,13:ulabel_dmp0,
    # 14-15:ulabel_dmp1,16-17:ulabel_dmp2,18-19:ulabel_inl,20:ulabel_cpd,21:date_wh,22:ulabel_titon),(23-24:ad1,25-26:ad2,27-28:ad3)

    def feature_columns(self, key, bucket_size, weighted=True, embedding=True):
        features = tf.feature_column.categorical_column_with_hash_bucket(
            key=key + "_features",
            hash_bucket_size=bucket_size,
            dtype=tf.string)
        if not weighted:
            return features
        weighted_features = tf.feature_column.weighted_categorical_column(
            categorical_column=features,
            weight_feature_key=key + "_weights",
            dtype=tf.float32)
        return weighted_features

    def build_feature_columns(self):
        user_adc_weighted_features = self.feature_columns("user_adc", bucket_size=200000)
        user_adoff_weighted_features = self.feature_columns("user_adoff", bucket_size=200000)
        user_news_weighted_features = self.feature_columns("user_news", bucket_size=20000)
        user_tt_weighted_features = self.feature_columns("user_tt", bucket_size=300000)
        user_gg_weighted_features = self.feature_columns("user_gg", bucket_size=10000)
        user_sea_features = self.feature_columns("user_sea", bucket_size=600000, weighted=False)
        user_dmp0_features = self.feature_columns("user_dmp0", bucket_size=10000, weighted=False)
        user_dmp1_weighted_features = self.feature_columns("user_dmp1", bucket_size=500)
        user_dmp2_weighted_features = self.feature_columns("user_dmp2", bucket_size=500)
        user_inl_weighted_features = self.feature_columns("user_inl", bucket_size=10000)
        user_cpd_features = self.feature_columns("user_cpd", bucket_size=10000, weighted=False)
        date_wh_features = self.feature_columns("date_wh", bucket_size=1000, weighted=False)
        user_titon_features = self.feature_columns("user_titon", bucket_size=300000, weighted=False)
        ad1_weighted_features = self.feature_columns("ad1", bucket_size=500)
        ad2_weighted_features = self.feature_columns("ad2", bucket_size=200000)
        ad3_weighted_features = self.feature_columns("ad3", bucket_size=500000)

    # def build_feature_columns(self):
    #     user_adc_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_adc_features",
    #         hash_bucket_size=20000,
    #         dtype=tf.string)
    #     user_adc_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_adc_features,
    #         weight_feature_key="user_adc_weights",
    #         dtype=tf.float32)
    #     user_adc_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_adc_weighted_features,
    #         dimension=64,
    #         combiner="sqrtn")
    #
    #     user_adoff_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_adoff_features",
    #         hash_bucket_size=20000,
    #         dtype=tf.string)
    #     user_adoff_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_adoff_features,
    #         weight_feature_key="user_adoff_weights",
    #         dtype=tf.float32)
    #     user_adoff_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_adoff_weighted_features,
    #         dimension=64,
    #         combiner="sqrtn")
    #
    #     user_news_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_news_features",
    #         hash_bucket_size=2000,
    #         dtype=tf.string)
    #     user_news_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_news_features,
    #         weight_feature_key="user_news_weights",
    #         dtype=tf.float32)
    #     user_news_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_news_weighted_features,
    #         dimension=64,
    #         combiner="sqrtn")
    #
    #     user_tt_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_tt_features",
    #         hash_bucket_size=30000,
    #         dtype=tf.string)
    #     user_tt_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_tt_features,
    #         weight_feature_key="user_tt_weights",
    #         dtype=tf.float32)
    #     user_tt_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_tt_weighted_features,
    #         dimension=128,
    #         combiner="sqrtn")
    #
    #     user_gg_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_gg_features",
    #         hash_bucket_size=1000,
    #         dtype=tf.string)
    #     user_gg_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_gg_features,
    #         weight_feature_key="user_gg_weights",
    #         dtype=tf.float32)
    #     user_gg_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_gg_weighted_features,
    #         dimension=64,
    #         combiner="sqrtn")
    #
    #     user_sea_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_sea_features",
    #         hash_bucket_size=60000,
    #         dtype=tf.string)
    #     user_sea_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_sea_features,
    #         dimension=128,
    #         combiner="sqrtn")
    #
    #     user_dmp0_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_dmp0_features",
    #         hash_bucket_size=1000,
    #         dtype=tf.string)
    #     user_dmp0_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_dmp0_features,
    #         dimension=16,
    #         combiner="sqrtn")
    #
    #     user_dmp1_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_dmp1_features",
    #         hash_bucket_size=50,
    #         dtype=tf.string)
    #     user_dmp1_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_dmp1_features,
    #         weight_feature_key="user_dmp1_weights",
    #         dtype=tf.float32)
    #     user_dmp1_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_dmp1_weighted_features,
    #         dimension=32,
    #         combiner="sqrtn")
    #
    #     user_dmp2_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_dmp2_features",
    #         hash_bucket_size=200,
    #         dtype=tf.string)
    #     user_dmp2_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_dmp2_features,
    #         weight_feature_key="user_dmp2_weights",
    #         dtype=tf.float32)
    #     user_dmp2_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_dmp2_weighted_features,
    #         dimension=32,
    #         combiner="sqrtn")
    #
    #     user_inl_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_inl_features",
    #         hash_bucket_size=1000,
    #         dtype=tf.string)
    #     user_inl_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=user_inl_features,
    #         weight_feature_key="user_inl_weights",
    #         dtype=tf.float32)
    #     user_inl_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_inl_weighted_features,
    #         dimension=64,
    #         combiner="sqrtn")
    #
    #     user_cpd_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_cpd_features",
    #         hash_bucket_size=1000,
    #         dtype=tf.string)
    #     user_cpd_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_cpd_features,
    #         dimension=32,
    #         combiner="sqrtn")
    #
    #     date_wh_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="date_wh_features",
    #         hash_bucket_size=100,
    #         dtype=tf.string)
    #     date_wh_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=date_wh_features,
    #         dimension=16,
    #         combiner="sqrtn")
    #
    #     user_titon_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="user_titon_features",
    #         hash_bucket_size=30000,
    #         dtype=tf.string)
    #     user_titon_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=user_titon_features,
    #         dimension=128,
    #         combiner="sqrtn")
    #
    #     ad1_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="ad1_features",
    #         hash_bucket_size=50,
    #         dtype=tf.string)
    #     ad1_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=ad1_features,
    #         weight_feature_key="ad1_weights",
    #         dtype=tf.float32)
    #     ad1_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=ad1_weighted_features,
    #         dimension=16,
    #         combiner="sqrtn")
    #
    #     ad2_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="ad2_features",
    #         hash_bucket_size=20000,
    #         dtype=tf.string)
    #     ad2_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=ad2_features,
    #         weight_feature_key="ad2_weights",
    #         dtype=tf.float32)
    #     ad2_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=ad2_weighted_features,
    #         dimension=128,
    #         combiner="sqrtn")
    #
    #     ad3_features = tf.feature_column.categorical_column_with_hash_bucket(
    #         key="ad3_features",
    #         hash_bucket_size=100000,
    #         dtype=tf.string)
    #     ad3_weighted_features = tf.feature_column.weighted_categorical_column(
    #         categorical_column=ad3_features,
    #         weight_feature_key="ad3_weights",
    #         dtype=tf.float32)
    #     ad3_embedding_features = tf.feature_column.embedding_column(
    #         categorical_column=ad3_weighted_features,
    #         dimension=128,
    #         combiner="sqrtn")

        linear_feature_columns = [ad1_weighted_features, ad2_weighted_features, ad3_weighted_features]
        dnn_feature_columns = [user_adc_embedding_features, user_adoff_embedding_features, user_news_embedding_features,
                               user_tt_embedding_features, user_gg_embedding_features, user_sea_embedding_features,
                               user_dmp0_embedding_features, user_dmp1_embedding_features, user_dmp2_embedding_features,
                               user_inl_embedding_features, user_cpd_embedding_features, date_wh_embedding_features,
                               user_titon_embedding_features, ad1_embedding_features, ad2_embedding_features,
                               ad3_embedding_features]

        user_feature_columns = [user_adc_embedding_features, user_adoff_embedding_features,
                                user_news_embedding_features, user_tt_embedding_features, user_gg_embedding_features,
                                user_sea_embedding_features, user_dmp0_embedding_features, user_dmp1_embedding_features,
                                user_dmp2_embedding_features, user_inl_embedding_features, user_cpd_embedding_features,
                                date_wh_embedding_features, user_titon_embedding_features]
        ads_feature_columns = [ad1_weighted_features, ad2_weighted_features, ad3_weighted_features,
                               ad1_embedding_features, ad2_embedding_features, ad3_embedding_features]

        return (linear_feature_columns, dnn_feature_columns, ads_feature_columns, user_feature_columns)

    # 1:click,(2-3:ulabel_adc,4-5:ulabel_adOff,6-7:ulabel_news,8-9:ulabel_tt,10-11:ulabel_gg,12:ulabel_ser,13:ulabel_dmp0,
    # 14-15:ulabel_dmp1,16-17:ulabel_dmp2,18-19:ulabel_inl,20:ulabel_cpd,21:date_wh,22:ulabel_titon),(23-24:ad1,25-26:ad2,27-28:ad3)

    def build_linear_optimizer(self):
        linear_optimizer = tf.train.FtrlOptimizer(
            learning_rate=0.05,
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.1,
            l2_regularization_strength=0.1)
        return linear_optimizer

    def build_dnn_optimizer(self):
        dnn_optimizer = tf.train.AdagradOptimizer(
            learning_rate=0.005,
            initial_accumulator_value=0.1)
        return dnn_optimizer

    def build_estimator(self):
        linear_optimizer = self.build_linear_optimizer()
        dnn_optimizer = self.build_dnn_optimizer()
        dnn_hidden_units = [512, 256, 128, 32]
        (linear_feature_columns, dnn_feature_columns, _, _) = self.build_feature_columns()

        if self.model_type == "wide":
            model = tf.estimator.LinearClassifier(
                feature_columns=linear_feature_columns,
                model_dir=self.model_dir,
                optimizer=linear_optimizer,
                config=self.run_config)
        elif self.model_type == "deep":
            model = tf.estimator.DNNClassifier(
                hidden_units=dnn_hidden_units,
                feature_columns=dnn_feature_columns,
                model_dir=self.model_dir,
                optimizer=dnn_optimizer,
                activation_fn=tf.nn.relu,
                config=self.run_config)
        elif self.model_type == "wide_and_deep":
            model = tf.estimator.DNNLinearCombinedClassifier(
                model_dir=self.model_dir,
                linear_feature_columns=linear_feature_columns,
                linear_optimizer=linear_optimizer,
                dnn_feature_columns=dnn_feature_columns,
                dnn_optimizer=dnn_optimizer,
                dnn_hidden_units=dnn_hidden_units,
                dnn_activation_fn=tf.nn.relu,
                config=self.run_config)
        else:
            logging.error("unsupported model type: %s" % (self.model_type))
        return model


####################################################################################################


class WideAndDeepInputPipeline(object):
    def __init__(self, input_files, batch_size=1000):
        self.batch_size = batch_size
        self.input_files = input_files

        input_file_list = []
        for input_file in self.input_files:
            if len(input_file) > 0:
                input_file_list.append(tf.train.match_filenames_once(input_file))
        self.filename_queue = tf.train.string_input_producer(
            tf.concat(input_file_list, axis=0),
            num_epochs=None,  # strings are repeated num_epochs
            shuffle=True,  # strings are randomly shuffled within each epoch
            capacity=512)
        self.reader = tf.TextLineReader(skip_header_lines=0)

        (self.column_dict, self.column_defaults) = self.build_column_format()

    # 1:click,(2-3:ulabel_adc,4-5:ulabel_adOff,6-7:ulabel_news,8-9:ulabel_tt,10-11:ulabel_gg,12:ulabel_ser,13:ulabel_dmp0,
    # 14-15:ulabel_dmp1,16-17:ulabel_dmp2,18-19:ulabel_inl,20:ulabel_cpd,21:date_wh,22:ulabel_titon),(23-24:ad1,25-26:ad2,27-28:ad3)

    def build_column_format(self):
        column_dict = {"label": 1, "user_adc_features": 1, "user_adc_weights": 2, "user_adoff_features": 3,
                       "user_adoff_weights": 4, "user_news_features": 5, "user_news_weights": 6, "user_tt_features": 7,
                       "user_tt_weights": 8, "user_gg_features": 9, "user_gg_weights": 10, "user_sea_features": 11,
                       "user_dmp0_features": 12, "user_dmp1_features": 13, "user_dmp1_weights": 14,
                       "user_dmp2_features": 15, "user_dmp2_weights": 16, "user_inl_features": 17,
                       "user_inl_weights": 18, "user_cpd_features": 19, "date_wh_features": 20,
                       "user_titon_features": 21, "ad1_features": 22, "ad1_weights": 23, "ad2_features": 24,
                       "ad2_weights": 25, "ad3_features": 26, "ad3_weights": 27}
        column_defaults = [['']] * len(column_dict)
        column_defaults[column_dict["label"]] = [0.0]
        column_defaults[column_dict["user_adc_features"]] = ['0']
        column_defaults[column_dict["user_adc_weights"]] = ['0.0']
        column_defaults[column_dict["user_adoff_features"]] = ['0']
        column_defaults[column_dict["user_adoff_weights"]] = ['0.0']
        column_defaults[column_dict["user_news_features"]] = ['0']
        column_defaults[column_dict["user_news_weights"]] = ['0.0']
        column_defaults[column_dict["user_tt_features"]] = ['0']
        column_defaults[column_dict["user_tt_weights"]] = ['0.0']
        column_defaults[column_dict["user_gg_features"]] = ['0']
        column_defaults[column_dict["user_gg_weights"]] = ['0.0']
        column_defaults[column_dict["user_sea_features"]] = ['0']
        column_defaults[column_dict["user_dmp0_features"]] = ['0']
        column_defaults[column_dict["user_dmp1_features"]] = ['0']
        column_defaults[column_dict["user_dmp1_weights"]] = ['0.0']
        column_defaults[column_dict["user_dmp2_features"]] = ['0']
        column_defaults[column_dict["user_dmp2_weights"]] = ['0.0']
        column_defaults[column_dict["user_inl_features"]] = ['0']
        column_defaults[column_dict["user_inl_weights"]] = ['0.0']
        column_defaults[column_dict["user_cpd_features"]] = ['0']
        column_defaults[column_dict["date_wh_features"]] = ['0']
        column_defaults[column_dict["user_titon_features"]] = ['0']
        column_defaults[column_dict["ad1_features"]] = ['0']
        column_defaults[column_dict["ad1_weights"]] = ['0.0']
        column_defaults[column_dict["ad2_features"]] = ['0']
        column_defaults[column_dict["ad2_weights"]] = ['0.0']
        column_defaults[column_dict["ad3_features"]] = ['0']
        column_defaults[column_dict["ad3_weights"]] = ['0.0']
        return (column_dict, column_defaults)

    def string_to_number(self, string_tensor, dtype=tf.float32):
        number_values = tf.string_to_number(
            string_tensor=string_tensor.values,
            out_type=dtype)
        number_tensor = tf.SparseTensor(
            indices=string_tensor.indices,
            values=number_values,
            dense_shape=string_tensor.dense_shape)
        return number_tensor

    def get_next_batch(self):
        (_, records) = self.reader.read_up_to(self.filename_queue, num_records=self.batch_size)
        samples = tf.decode_csv(records, record_defaults=self.column_defaults, field_delim=',')
        label = tf.cast(samples[self.column_dict["label"]], dtype=tf.int32)
        feature_dict = {}
        for (key, value) in self.column_dict.items():
            if key == "label" or value < 0 or value >= len(samples):
                continue
            if key in ["user_adc_features", "user_adoff_features", "user_news_features", "user_tt_features",
                       "user_gg_features", "user_sea_features", "user_dmp0_features", "user_dmp1_features",
                       "user_dmp2_features", "user_inl_features", "user_cpd_features", "date_wh_features",
                       "user_titon_features", "ad1_features", "ad2_features", "ad3_features"]:
                feature_dict[key] = tf.string_split(samples[value], delimiter=';')
            if key in ["user_adc_weights", "user_adoff_weights", "user_news_weights", "user_tt_weights",
                       "user_gg_weights", "user_dmp1_weights", "user_dmp2_weights", "user_inl_weights", "ad1_weights",
                       "ad2_weights", "ad3_weights"]:
                feature_dict[key] = self.string_to_number(
                    tf.string_split(samples[value], delimiter=';'),
                    dtype=tf.float32)
        return feature_dict, label


####################################################################################################


def train_input_fn():
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = WideAndDeepInputPipeline(
        train_input_files,
        batch_size=FLAGS.batch_size)
    return train_input_pipeline.get_next_batch()


def train_model():
    if FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        # tf.gfile.DeleteRecursively(FLAGS.model_dir)
        pass

    model = WideAndDeepModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size)
    estimator = model.build_estimator()
    estimator.train(
        input_fn=lambda: train_input_fn(),
        steps=100000)


def eval_input_fn():
    eval_input_files = FLAGS.eval_data.strip().split(',')
    eval_input_pipeline = WideAndDeepInputPipeline(
        eval_input_files,
        batch_size=FLAGS.batch_size)
    return eval_input_pipeline.get_next_batch()


def eval_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    # Get the checkpoint path for evaluation
    checkpoint_path = None  # The latest checkpoint in model dir
    if FLAGS.eval_ckpt_id > 0:
        state = tf.train.get_checkpoint_state(
            checkpoint_dir=FLAGS.model_dir,
            latest_filename="checkpoint")
        if state and state.all_model_checkpoint_paths:
            if FLAGS.eval_ckpt_id < len(state.all_model_checkpoint_paths):
                pos = -(1 + FLAGS.eval_ckpt_id)
                checkpoint_path = state.all_model_checkpoint_paths[pos]
            else:
                logging.warn("not find checkpoint id %d in %s" % (FLAGS.eval_ckpt_id, FLAGS.model_dir))
                checkpoint_path = None
    logging.info("checkpoint path: %s" % (checkpoint_path))
    eval_name = '' if checkpoint_path is None else str(FLAGS.eval_ckpt_id)

    model = WideAndDeepModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size)
    estimator = model.build_estimator()
    eval_result = estimator.evaluate(
        input_fn=lambda: eval_input_fn(),
        steps=100,
        checkpoint_path=checkpoint_path,
        name=eval_name)
    print(eval_result)


def build_custom_serving_input_receiver_fn(ads_feature_spec, user_feature_spec):
    def serving_input_receiver_fn():
        ads_serialized_examples = array_ops.placeholder(dtype=dtypes.string)
        user_serialized_example = array_ops.placeholder(dtype=dtypes.string)
        receiver_tensors = {'ads_data': ads_serialized_examples, 'user_data': user_serialized_example}
        user_serialized_examples = tf.tile(user_serialized_example, [tf.shape(ads_serialized_examples)[0]])
        ads_features = parsing_ops.parse_example(ads_serialized_examples, ads_feature_spec)
        user_features = parsing_ops.parse_example(user_serialized_examples, user_feature_spec)
        user_features.update(ads_features)
        return ServingInputReceiver(user_features, receiver_tensors)

    return serving_input_receiver_fn


def export_savedmodel():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    model = WideAndDeepModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size)
    estimator = model.build_estimator()

    if FLAGS.savedmodel_mode == "raw":
        features = model.build_feature_dict()
        export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            features=features,
            default_batch_size=None)
    elif FLAGS.savedmodel_mode == "parsing":
        (linear_feature_columns, dnn_feature_columns, _, _) = model.build_feature_columns()
        feature_columns = linear_feature_columns + dnn_feature_columns
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec=feature_spec,
            default_batch_size=None)
    elif FLAGS.savedmodel_mode == "parsing_new":
        (_, _, ads_feature_columns, user_feature_columns) = model.build_feature_columns()
        ads_feature_spec = tf.feature_column.make_parse_example_spec(ads_feature_columns)
        user_feature_spec = tf.feature_column.make_parse_example_spec(user_feature_columns)
        export_input_fn = build_custom_serving_input_receiver_fn(ads_feature_spec, user_feature_spec)
    else:
        logging.error("unsupported savedmodel mode: %s" % (FLAGS.savedmodel_mode))
        sys.exit(1)

    export_dir = estimator.export_savedmodel(
        export_dir_base=FLAGS.export_savedmodel,
        serving_input_receiver_fn=lambda: export_input_fn(),
        assets_extra=None,
        as_text=False,
        checkpoint_path=None)


def predict_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    model = WideAndDeepModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size)
    estimator = model.build_estimator()
    predict = estimator.predict(
        input_fn=lambda: eval_input_fn(),
        predict_keys=None,
        hooks=None,
        checkpoint_path=None)
    results = list(predict)
    sum_score = 0.0
    for i in range(0, len(results)):
        result = results[i]
        sum_score = sum_score + result["logistic"][0]
        print("count: %d, score: %f" % (i + 1, result["logistic"][0]))
    print("total count: %d, average score: %f" % (len(results), sum_score / len(results)))


####################################################################################################


def test_input_pipeline():
    logging.info("train data: %s" % (FLAGS.train_data))
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = WideAndDeepInputPipeline(
        train_input_files,
        batch_size=3)  # FLAGS.batch_size)
    feature_batch, label_batch = train_input_pipeline.get_next_batch()
    # print(label_batch)
    # print(feature_batch)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        features, labels = sess.run([feature_batch, label_batch])
        print(labels)
        print(features["user_features"])
        print(features["user_features"].dense_shape)
        print(features["user_weights"].dense_shape)
        print(features["ads_features"].dense_shape)
        print(features["ads_weights"].dense_shape)
        coord.request_stop()
        coord.join(threads)


####################################################################################################

class InnerExperiment(tf.contrib.learn.Experiment):  # BTBT
    def __init__(self,
                 estimator,
                 train_input_fn,
                 eval_input_fn,
                 eval_metrics=None,
                 train_steps=None,
                 eval_steps=100,
                 train_monitors=None,
                 eval_hooks=None,
                 local_eval_frequency=None,
                 eval_delay_secs=120,
                 continuous_eval_throttle_secs=60,
                 min_eval_frequency=None,
                 delay_workers_by_global_step=False,
                 export_strategies=None,
                 train_steps_per_iteration=None,
                 checkpoint_and_export=False):
        super(InnerExperiment, self).__init__(estimator=estimator,
                                              train_input_fn=train_input_fn,
                                              eval_input_fn=eval_input_fn,
                                              eval_metrics=eval_metrics,
                                              train_steps=train_steps,
                                              eval_steps=eval_steps,
                                              train_monitors=train_monitors,
                                              eval_hooks=eval_hooks,
                                              local_eval_frequency=local_eval_frequency,
                                              eval_delay_secs=eval_delay_secs,
                                              continuous_eval_throttle_secs=continuous_eval_throttle_secs,
                                              min_eval_frequency=min_eval_frequency,
                                              delay_workers_by_global_step=delay_workers_by_global_step,
                                              export_strategies=export_strategies,
                                              train_steps_per_iteration=train_steps_per_iteration,
                                              checkpoint_and_export=checkpoint_and_export)

    def _call_train(self, _sentinel=None,  # pylint: disable=invalid-name,
                    input_fn=None, steps=None, hooks=None, max_steps=None,
                    saving_listeners=None):
        logging.info("***** INNER._call_train *****")
        if _sentinel is not None:
            raise ValueError("_call_train should be called with keyword args only")

        # Estimator in core cannot work with monitors. We need to convert them
        # to hooks. For Estimator in contrib, it is converted internally. So, it is
        # safe to convert for both cases.
        hooks = monitors.replace_monitors_with_hooks(hooks, self._estimator)
        if self._core_estimator_used:
            return self._estimator.train(
                input_fn=input_fn,
                steps=max_steps,  # the train_step
                max_steps=None,
                hooks=hooks,
                saving_listeners=saving_listeners)
        else:
            return self._estimator.fit(input_fn=input_fn,
                                       steps=max_steps,
                                       max_steps=None,
                                       monitors=hooks)


class WideAndDeepDistributedRunner(object):
    def __init__(self,
                 model_type="wide",
                 model_dir=None,
                 feature_size=10000000,
                 schedule="train",
                 worker_hosts=None,
                 ps_hosts=None,
                 task_type=None,
                 task_index=None):
        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        self.schedule = schedule
        self.worker_hosts = worker_hosts.strip().split(",")
        self.ps_hosts = ps_hosts.strip().split(",")
        self.task_type = task_type
        self.task_index = task_index

        self.run_config = self.build_run_config()
        self.hparams = self.build_hparams()

    def build_run_config(self):
        cluster = {"worker": self.worker_hosts, "ps": self.ps_hosts}
        task = {"type": self.task_type, "index": self.task_index}
        environment = {"environment": "cloud"}
        os.environ["TF_CONFIG"] = json.dumps({"cluster": cluster, "task": task, "environment": environment})

        run_config = tf.contrib.learn.RunConfig(
            tf_random_seed=1,  # Random seed for TensorFlow initializers
            save_summary_steps=1000,  # Save summaries every this many steps
            save_checkpoints_secs=None,  # Save checkpoints every this many seconds
            save_checkpoints_steps=100,  # Save checkpoints every this many steps
            keep_checkpoint_max=5,  # The maximum number of recent checkpoint files to keep
            keep_checkpoint_every_n_hours=10000,  # Number of hours between each checkpoint to be saved
            log_step_count_steps=100,  # The frequency, in number of global steps
            model_dir=self.model_dir,  # Directory where model parameters, graph etc are saved
            session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                                          device_filters=["/job:ps", "/job:worker/task:%d" % self.task_index])  # BTBT
        )
        return run_config

    def build_hparams(self):
        hparams = tf.contrib.training.HParams(
            eval_metrics=None,
            train_steps=None,
            eval_steps=100,
            eval_delay_secs=5,
            min_eval_frequency=None)
        return hparams

    def build_experiment(self, run_config, hparams):
        model = WideAndDeepModel(
            model_type=self.model_type,
            model_dir=self.model_dir,
            feature_size=self.feature_size,
            run_config=run_config)
        return InnerExperiment(  # BTBT
            estimator=model.build_estimator(),
            train_input_fn=lambda: train_input_fn(),
            eval_input_fn=lambda: eval_input_fn(),
            eval_metrics=hparams.eval_metrics,
            train_steps=hparams.train_steps,
            eval_steps=hparams.eval_steps,
            eval_delay_secs=hparams.eval_delay_secs,
            min_eval_frequency=hparams.min_eval_frequency)

    def run(self):
        tf.contrib.learn.learn_runner.run(
            experiment_fn=self.build_experiment,
            output_dir=None,  # Deprecated, must be None
            schedule=self.schedule,
            run_config=self.run_config,
            hparams=self.hparams)


####################################################################################################


def local_run():
    if FLAGS.job_type == "train":
        train_model()
    elif FLAGS.job_type == "eval":
        eval_model()
    elif FLAGS.job_type == "train_and_eval":
        train_model()
        eval_model()
    elif FLAGS.job_type == "export_savedmodel":
        export_savedmodel()
    elif FLAGS.job_type == "predict":
        predict_model()
    else:
        logging.error("unsupported job type: %s" % (FLAGS.job_type))
        sys.exit(1)


def distributed_run():
    if FLAGS.task_type == "worker" and FLAGS.task_index == 0 \
            and (FLAGS.job_type == "train" or FLAGS.job_type == "train_and_eval") \
            and FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        # tf.gfile.DeleteRecursively(FLAGS.model_dir)
        pass

    schedule = None
    schedule_dict = {"train": "train", "eval": "evaluate", "train_and_eval": "train_and_evaluate"}
    if FLAGS.task_type == "ps":
        schedule = "run_std_server"
    elif FLAGS.task_type == "worker":
        schedule = schedule_dict.get(FLAGS.job_type, None)
        if FLAGS.job_type == "train_and_eval" and FLAGS.task_index != 0:
            schedule = "train"  # only the first worker runs evaluation
        if FLAGS.task_index == 0:
            logging.error("**** Worder 0 sleep for 10 sec begin")
            time.sleep(10)
            logging.error("**** Worder 0 sleep for 10 sec end")
    else:
        logging.error("unsupported task type: %s" % (FLAGS.task_type))
        sys.exit(1)
    logging.info("schedule: %s" % (schedule))

    runner = WideAndDeepDistributedRunner(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size,
        schedule=schedule,
        worker_hosts=FLAGS.worker_hosts,
        ps_hosts=FLAGS.ps_hosts,
        task_type=FLAGS.task_type,
        task_index=FLAGS.task_index)
    runner.run()


def main():
    # print commandline arguments
    logging.info("run mode: %s" % (FLAGS.run_mode))
    if FLAGS.run_mode == "distributed":
        logging.info("worker hosts: %s" % (FLAGS.worker_hosts))
        logging.info("ps hosts: %s" % (FLAGS.ps_hosts))
        logging.info("task type: %s, task index: %d" % (FLAGS.task_type, FLAGS.task_index))
    logging.info("job type: %s" % (FLAGS.job_type))
    if FLAGS.job_type == "train" or FLAGS.job_type == "train_and_eval":
        logging.info("train data: %s" % (FLAGS.train_data))
        logging.info("cold start: %s" % (FLAGS.cold_start))
    if FLAGS.job_type in ["eval", "train_and_eval", "predict"]:
        logging.info("eval data: %s" % (FLAGS.eval_data))
        logging.info("eval ckpt id: %s" % (FLAGS.eval_ckpt_id))
    if FLAGS.job_type == "export_savedmodel":
        logging.info("export savedmodel: %s" % (FLAGS.export_savedmodel))
        logging.info("savedmodel mode: %s" % (FLAGS.savedmodel_mode))
    logging.info("model dir: %s" % (FLAGS.model_dir))
    logging.info("model type: %s" % (FLAGS.model_type))
    logging.info("feature size: %s" % (FLAGS.feature_size))
    logging.info("batch size: %s" % (FLAGS.batch_size))

    if FLAGS.run_mode == "local":
        local_run()
    elif FLAGS.run_mode == "distributed":
        if FLAGS.job_type == "export_savedmodel" or FLAGS.job_type == "predict":
            logging.error("job type export_savedmodel and predict does not support distributed run mode")
            sys.exit(1)
        if FLAGS.job_type in ["eval", "train_and_eval"] and FLAGS.eval_ckpt_id != 0:
            logging.error("eval_ckpt_id does not support distributed run mode")
            sys.exit(1)
        distributed_run()
    else:
        logging.error("unsupported run mode: %s" % (FLAGS.run_mode))
        sys.exit(1)
    prefix = "" if FLAGS.run_mode == "local" else "%s:%d " % (FLAGS.task_type, FLAGS.task_index)
    logging.info("%scompleted" % (prefix))


def test():
    test_input_pipeline()


if __name__ == "__main__":
    main()
    # test()

