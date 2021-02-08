# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
import json
import time
from code import feature_column
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import monitors
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
flags.DEFINE_string("job_type", "train", "the job type, one of train, eval, train_and_eval, export_savedmodel and predict")

flags.DEFINE_string("train_data", "./train_data", "the train data, delimited by comma")
flags.DEFINE_string("eval_data", "./eval_data", "the eval data, delimited by comma")
flags.DEFINE_integer("batch_size", 1000, "the batch size")
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

# hyper parameters
flags.DEFINE_float("linear_learning_rate", 0.05, "linear_learning_rate")
flags.DEFINE_float("dnn_learning_rate", 0.05, "dnn_learning_rate")
flags.DEFINE_string("dnn_hidden_units", "256-128-32", "dnn_hidden_units")

flags.DEFINE_string("feature_map_file", "./feature_map_v31_v1.json", "path of feature_map")


class WideAndDeepModel(object):
    def __init__(self,
            model_type = "wide",
            model_dir = None,
            feature_size = 100000,
            run_config = None,
            feature_map_file = "./feature_map.json"):

        self.model_type = model_type
        self.model_dir = model_dir
        self.feature_size = feature_size
        with open(feature_map_file, "r") as f:
            self.feature_info = json.load(f)

        if run_config is None:
            self.run_config = self.build_run_config()
        else:
            self.run_config = run_config


    def build_run_config(self):
        run_config = tf.contrib.learn.RunConfig(
                tf_random_seed = 1,                     # Random seed for TensorFlow initializers
                save_summary_steps = 10,               # Save summaries every this many steps
                save_checkpoints_secs = 300,            # Save checkpoints every this many seconds
                save_checkpoints_steps = None,          # Save checkpoints every this many steps
                keep_checkpoint_max = 100,                # The maximum number of recent checkpoint files to keep
                keep_checkpoint_every_n_hours = 10000,  # Number of hours between each checkpoint to be saved
                log_step_count_steps = 10)             # The frequency, in number of global steps
        return run_config


    def build_feature_dict(self):
        feature_dict = {}
        for key in self.feature_info["concat_features"]:
            feature_dict[key] = tf.placeholder(dtype = tf.string, shape = (None, self.feature_size))
        
        for key in self.feature_info["concat_weights"]:
            feature_dict[key] = tf.placeholder(dtype = tf.float32, shape = (None, self.feature_size))

        return feature_dict

    def build_feature_columns(self):
        feature_hash_bucket_dict = {}
        feature_weighted_categorical_column_dict = {}
        feature_embedding_column_dcit = {}
        
        for index, feature in enumerate(self.feature_info["concat_features"]):
            feature_weight_name = feature.replace("features", "weights") 
            feature_hash_bucket_dict[feature] = tf.feature_column.categorical_column_with_hash_bucket(
                    key=feature,
                    hash_bucket_size=self.feature_info["hash_bucket_size"][index],
                    dtype=tf.string)

            if feature_weight_name in self.feature_info["concat_weights"]:
                feature_weighted_categorical_column_dict[feature] = tf.feature_column.weighted_categorical_column(
                        categorical_column=feature_hash_bucket_dict[feature],
                        weight_feature_key=feature_weight_name,
                        dtype=tf.float32)

                if feature in self.feature_info["indicator_columns"]:
                    feature_embedding_column_dcit[feature] = feature_column.indicator_columnV1(feature_weighted_categorical_column_dict[feature])
                else:
                    feature_embedding_column_dcit[feature] = tf.feature_column.embedding_column(
                            categorical_column=feature_weighted_categorical_column_dict[feature],
                            dimension=self.feature_info["embedding_dimension"][index],
                            combiner = "sqrtn")
            
            if feature_weight_name not in self.feature_info["concat_weights"]:
                feature_embedding_column_dcit[feature] = tf.feature_column.embedding_column(
                        categorical_column=feature_hash_bucket_dict[feature],
                        dimension=self.feature_info["embedding_dimension"][index],
                        combiner = "sqrtn")

        linear_feature_columns = []
        dnn_feature_columns = []
        user_feature_columns = []
        ads_feature_columns = []

        for index, feature in enumerate(self.feature_info["linear_feature_columns"]):
            linear_feature_columns.append(feature_weighted_categorical_column_dict[feature])

        for index, feature in enumerate(self.feature_info["dnn_feature_columns"]):
            dnn_feature_columns.append(feature_embedding_column_dcit[feature])
        
        for index, feature in enumerate(self.feature_info["user_feature_columns"]):
            user_feature_columns.append(feature_embedding_column_dcit[feature])

        for index, feature in enumerate(self.feature_info["ads_feature_columns"]):
            ads_feature_columns.append(feature_embedding_column_dcit[feature])
            ads_feature_columns.append(feature_weighted_categorical_column_dict[feature])

        return (linear_feature_columns, dnn_feature_columns, ads_feature_columns, user_feature_columns)

    def build_linear_optimizer(self):
        linear_optimizer = tf.train.FtrlOptimizer(
                learning_rate = float(FLAGS.linear_learning_rate),
                learning_rate_power = -0.5,
                initial_accumulator_value = 0.1,
                l1_regularization_strength = 0.1,
                l2_regularization_strength = 0.1)
        return linear_optimizer


    def build_dnn_optimizer(self):
        dnn_optimizer = tf.train.AdagradOptimizer(
                learning_rate = float(FLAGS.dnn_learning_rate),
                initial_accumulator_value = 0.1)
        return dnn_optimizer


    def build_estimator(self):
        linear_optimizer = self.build_linear_optimizer()
        dnn_optimizer = self.build_dnn_optimizer()
        dnn_hidden_units = [int(i) for i in FLAGS.dnn_hidden_units.split("-")]
        (linear_feature_columns, dnn_feature_columns,_,_) = self.build_feature_columns()

        if self.model_type == "wide":
            model = tf.estimator.LinearClassifier(
                    feature_columns = linear_feature_columns,
                    model_dir = self.model_dir,
                    optimizer = linear_optimizer,
                    config = self.run_config)
        elif self.model_type == "deep":
            model = tf.estimator.DNNClassifier(
                    hidden_units = dnn_hidden_units,
                    feature_columns = dnn_feature_columns,
                    model_dir = self.model_dir,
                    optimizer = dnn_optimizer,
                    activation_fn = tf.nn.relu,
                    config = self.run_config)
        elif self.model_type == "wide_and_deep":
            model = tf.estimator.DNNLinearCombinedClassifier(
                    model_dir = self.model_dir,
                    linear_feature_columns = linear_feature_columns,
                    linear_optimizer = linear_optimizer,
                    dnn_feature_columns = dnn_feature_columns,
                    dnn_optimizer = dnn_optimizer,
                    dnn_hidden_units = dnn_hidden_units,
                    dnn_activation_fn = tf.nn.relu,
                    config = self.run_config)
        else:
            logging.error("unsupported model type: %s" % (self.model_type))
        return model


####################################################################################################


class WideAndDeepInputPipeline(object):
    def __init__(self, input_files, batch_size = 1000, feature_map_file = "./feature_map.json"):
        self.batch_size = batch_size
        self.input_files = input_files
        
        with open(feature_map_file, "r") as f:
            self.feature_info = json.load(f)

        input_file_list = []
        for input_file in self.input_files:
            if len(input_file) > 0:
                input_file_list.append(tf.train.match_filenames_once(input_file))
        self.filename_queue = tf.train.string_input_producer(
                tf.concat(input_file_list, axis = 0),
                num_epochs = 20,
                shuffle = True,
                capacity = 512)
        self.reader = tf.TextLineReader(skip_header_lines = 0)

        (self.column_dict, self.column_defaults) = self.build_column_format()

    def build_column_format(self):
        column_list = self.feature_info["columns"]
        column_dict = {field: index for index, field in enumerate(column_list)}

        column_defaults = [['']] * len(column_dict)
        for col in column_list:
            if col == "label":
                column_defaults[column_dict[col]] = [0.0]
            if "weights" in col:
                column_defaults[column_dict[col]] = ['0.0']
            if "weights" not in col and col not in ["request_ID", "label"]:
                column_defaults[column_dict[col]] = ['0']

        return column_dict, column_defaults

    def string_to_number(self, string_tensor, dtype = tf.float32):
        number_values = tf.string_to_number(
                string_tensor = string_tensor.values,
                out_type = dtype)
        number_tensor = tf.SparseTensor(
                indices = string_tensor.indices,
                values = number_values,
                dense_shape = string_tensor.dense_shape)
        return number_tensor


    def get_next_batch(self):
        (_, records) = self.reader.read_up_to(self.filename_queue, num_records = self.batch_size)
        samples = tf.decode_csv(records, record_defaults = self.column_defaults, field_delim = ',')
        label = tf.cast(samples[self.column_dict["label"]], dtype = tf.int32)
        feature_dict = {}
        feature_dict_concat = {}
        concat_features = []
        concat_weights = []
        
        for key, value in self.column_dict.items():
            if key == "label" or value < 0 or value >= len(samples):
                continue
            if key in self.feature_info["selected_features"]:
                feature_dict[key] = tf.string_split(samples[value], delimiter = ';')
            if key in self.feature_info["selected_weights"]:
                feature_dict[key] = self.string_to_number(tf.string_split(samples[value], delimiter = ';'), dtype = tf.float32)

        for combined_features, list_features in self.feature_info["combined_features"].items():
            concat_features.extend(list_features)
            feature_dict_concat[combined_features] = tf.sparse_concat(axis=1,
                    sp_inputs=[feature_dict[key] for key in list_features])
        
        for combined_weights, list_weights in self.feature_info["combined_weights"].items():
            concat_weights.extend(list_weights)
            feature_dict_concat[combined_weights] = tf.sparse_concat(axis=1,
                    sp_inputs=[feature_dict[key] for key in list_weights])

        for item in self.feature_info["selected_features"]:
            if item not in concat_features:
                feature_dict_concat[item] = feature_dict[item]
        
        for item in self.feature_info["selected_weights"]:
            if item not in concat_weights:
                feature_dict_concat[item] = feature_dict[item]


        return feature_dict_concat, label


####################################################################################################


def train_input_fn():
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = WideAndDeepInputPipeline(
            train_input_files,
            batch_size = FLAGS.batch_size,
            feature_map_file = FLAGS.feature_map_file)
    return train_input_pipeline.get_next_batch()


def train_model():
    if FLAGS.cold_start and tf.gfile.Exists(FLAGS.model_dir):
        #tf.gfile.DeleteRecursively(FLAGS.model_dir)
        pass

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            run_config = None,
            feature_map_file = FLAGS.feature_map_file)
    estimator = model.build_estimator()
    profile_hook = [tf.train.ProfilerHook(save_steps=1000)]
    estimator.train(
            input_fn = lambda: train_input_fn(),
            steps = 100000,
            hooks = profile_hook)


def eval_input_fn():
    eval_input_files = FLAGS.eval_data.strip().split(',')
    eval_input_pipeline = WideAndDeepInputPipeline(
            eval_input_files,
            batch_size = FLAGS.batch_size,
            feature_map_file = FLAGS.feature_map_file)
    return eval_input_pipeline.get_next_batch()


def eval_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    # Get the checkpoint path for evaluation
    checkpoint_path = None # The latest checkpoint in model dir
    if FLAGS.eval_ckpt_id > 0:
        state = tf.train.get_checkpoint_state(
                checkpoint_dir = FLAGS.model_dir,
                latest_filename = "checkpoint")
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
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            run_config = None,
            feature_map_file = FLAGS.feature_map_file)
    estimator = model.build_estimator()
    eval_result = estimator.evaluate(
            input_fn = lambda: eval_input_fn(),
            steps = 100,
            checkpoint_path = checkpoint_path,
            name = eval_name)
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
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            run_config = None,
            feature_map_file = FLAGS.feature_map_file)
    estimator = model.build_estimator()

    if FLAGS.savedmodel_mode == "raw":
        features = model.build_feature_dict()
        export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
                features = features,
                default_batch_size = None)
    elif FLAGS.savedmodel_mode == "parsing":
        (linear_feature_columns, dnn_feature_columns,_,_) = model.build_feature_columns()
        feature_columns = linear_feature_columns + dnn_feature_columns
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
                feature_spec = feature_spec,
                default_batch_size = None)
    elif FLAGS.savedmodel_mode == "parsing_new":
        (_,_,ads_feature_columns, user_feature_columns) = model.build_feature_columns()
        ads_feature_spec = tf.feature_column.make_parse_example_spec(ads_feature_columns)
        user_feature_spec = tf.feature_column.make_parse_example_spec(user_feature_columns)
        export_input_fn = build_custom_serving_input_receiver_fn(ads_feature_spec, user_feature_spec)
    else:
        logging.error("unsupported savedmodel mode: %s" % (FLAGS.savedmodel_mode))
        sys.exit(1)

    export_dir = estimator.export_savedmodel(
            export_dir_base = FLAGS.export_savedmodel,
            serving_input_receiver_fn = lambda: export_input_fn(),
            assets_extra = None,
            as_text = False,
            checkpoint_path = None)


def predict_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    model = WideAndDeepModel(
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            run_config = None,
            feature_map_file = FLAGS.feature_map_file)
    estimator = model.build_estimator()
    predict = estimator.predict(
            input_fn = lambda: eval_input_fn(),
            predict_keys = None,
            hooks = None,
            checkpoint_path = None)
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
            batch_size = 1,
            feature_map_file = FLAGS.feature_map_file) #FLAGS.batch_size)
    feature_batch, label_batch = train_input_pipeline.get_next_batch()
    #print(label_batch)
    #print(feature_batch)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        features, labels = sess.run([feature_batch, label_batch])
        
        import pdb
        pdb.set_trace()

        print(labels)
        # print(features["user_features"])
        # print(features["user_features"].dense_shape)
        # print(features["user_weights"].dense_shape)
        # print(features["ads_features"].dense_shape)
        # print(features["ads_weights"].dense_shape)
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
          steps=max_steps,#the train_step
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
            model_type = "wide",
            model_dir = None,
            feature_size = 10000000,
            schedule = "train",
            worker_hosts = None,
            ps_hosts = None,
            task_type = None,
            task_index = None):
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
                tf_random_seed = 1,                     # Random seed for TensorFlow initializers
                save_summary_steps = 100,              # Save summaries every this many steps
                save_checkpoints_secs = 120,            # Save checkpoints every this many seconds
                save_checkpoints_steps = None,          # Save checkpoints every this many steps
                keep_checkpoint_max = 100,                # The maximum number of recent checkpoint files to keep
                keep_checkpoint_every_n_hours = 10000,  # Number of hours between each checkpoint to be saved
                log_step_count_steps = 10,            # The frequency, in number of global steps
                model_dir = self.model_dir,             # Directory where model parameters, graph etc are saved
                session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False,
                        device_filters=["/job:ps", "/job:worker/task:%d" % self.task_index])  # BTBT
        )
        return run_config


    def build_hparams(self):
        hparams = tf.contrib.training.HParams(
                eval_metrics = None,
                train_steps = None,
                eval_steps = 100,
                eval_delay_secs = 5,
                min_eval_frequency = None)
        return hparams


    def build_experiment(self, run_config, hparams):
        model = WideAndDeepModel(
                model_type = self.model_type,
                model_dir = self.model_dir,
                feature_size = self.feature_size,
                run_config = run_config,
                feature_map_file = FLAGS.feature_map_file)
        return InnerExperiment(   # BTBT
                estimator = model.build_estimator(),
                train_input_fn = lambda: train_input_fn(),
                eval_input_fn = lambda: eval_input_fn(),
                eval_metrics = hparams.eval_metrics,
                train_steps = hparams.train_steps,
                eval_steps = hparams.eval_steps,
                eval_delay_secs = hparams.eval_delay_secs,
                min_eval_frequency = hparams.min_eval_frequency)


    def run(self):
        tf.contrib.learn.learn_runner.run(
                experiment_fn = self.build_experiment,
                output_dir = None, # Deprecated, must be None
                schedule = self.schedule,
                run_config = self.run_config,
                hparams = self.hparams)


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
        #tf.gfile.DeleteRecursively(FLAGS.model_dir)
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
            model_type = FLAGS.model_type,
            model_dir = FLAGS.model_dir,
            feature_size = FLAGS.feature_size,
            schedule = schedule,
            worker_hosts = FLAGS.worker_hosts,
            ps_hosts = FLAGS.ps_hosts,
            task_type = FLAGS.task_type,
            task_index = FLAGS.task_index)
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
    logging.info("linear_learning_rate: %s" % FLAGS.linear_learning_rate)
    logging.info("dnn_learning_rate: %s" % FLAGS.dnn_learning_rate)
    logging.info("dnn_hidden_units: %s" % FLAGS.dnn_hidden_units)
    logging.info("feature_map_file: %s" % FLAGS.feature_map_file)


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
    #test()

