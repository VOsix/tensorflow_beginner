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

flags.DEFINE_string("train_data", "../../data_taobao/taobao_train.csv", "the train data, delimited by comma")
flags.DEFINE_string("eval_data", "../../data_taobao/taobao_test.csv", "the eval data, delimited by comma")
flags.DEFINE_string("model_type", "wide", "the model type, one of wide, deep and wide_and_deep")
flags.DEFINE_string("job_type", "export_savedmodel",
                    "the job type, one of train, eval, train_and_eval, export_savedmodel and predict")
flags.DEFINE_string("model_dir", "./model_dir", "the model dir")
flags.DEFINE_bool("cold_start", False, "True: cold start; False: start from the latest checkpoint")
flags.DEFINE_integer("feature_size", 100, "the feature size")
flags.DEFINE_integer("batch_size", 1000, "the batch size")
flags.DEFINE_integer("num_epochs", 30, "the epoch num")
flags.DEFINE_float("linear_learning_rate", 0.05, "linear_learning_rate")
flags.DEFINE_float("dnn_learning_rate", 0.05, "dnn_learning_rate")
flags.DEFINE_string("dnn_hidden_units", "256-128-32", "dnn_hidden_units")
flags.DEFINE_integer("eval_ckpt_id", 0, "the checkpoint id in model dir for evaluation, 0 is the latest checkpoint")
flags.DEFINE_string("feature_map_file", "./feature_map.json", "path of feature_map")
flags.DEFINE_string("export_savedmodel", "./export_model", "the export savedmodel directory, used for tf serving")
flags.DEFINE_string("savedmodel_mode", "raw", "the savedmodel mode, one of raw and parsing")


class LogisticModel(object):
    def __init__(self,
                 model_dir=None,
                 model_type="wide",
                 feature_size=100000,
                 run_config=None,
                 feature_map_file="./feature_map.json"):

        self.model_dir = model_dir
        self.model_type = model_type
        self.feature_size = feature_size
        with open(feature_map_file, "r") as f:
            self.feature_info = json.load(f)

        if run_config is None:
            self.run_config = self.build_run_config()
        else:
            self.run_config = run_config

    def build_run_config(self):
        run_config = tf.contrib.learn.RunConfig(
            tf_random_seed=1,  # Random seed for TensorFlow initializers
            save_summary_steps=100,  # Save summaries every this many steps
            save_checkpoints_secs=300,  # Save checkpoints every this many seconds
            save_checkpoints_steps=None,  # Save checkpoints every this many steps
            keep_checkpoint_max=100,  # The maximum number of recent checkpoint files to keep
            keep_checkpoint_every_n_hours=10000,  # Number of hours between each checkpoint to be saved
            log_step_count_steps=100)  # The frequency, in number of global steps
        return run_config

    def build_feature_dict(self):
        feature_dict = {}
        for key in self.feature_info["selected_features"]:
            feature_dict[key] = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_size))

        return feature_dict

    def build_feature_columns(self):
        feature_numeric_dict = {}

        for index, feature in enumerate(self.feature_info["selected_features"]):
            feature_numeric_dict[feature] = tf.feature_column.numeric_column(
                key=feature,
                dtype=tf.float32
            )

        linear_feature_columns = []
        dnn_feature_columns = []

        for index, feature in enumerate(self.feature_info["linear_feature_columns"]):
            linear_feature_columns.append(feature_numeric_dict[feature])

        for index, feature in enumerate(self.feature_info["dnn_feature_columns"]):
            dnn_feature_columns.append(feature_numeric_dict[feature])

        return linear_feature_columns, dnn_feature_columns

    def build_linear_optimizer(self):
        # linear_optimizer = tf.train.FtrlOptimizer(
        #     learning_rate=float(FLAGS.linear_learning_rate),
        #     learning_rate_power=-0.5,
        #     initial_accumulator_value=0.1,
        #     l1_regularization_strength=0.1,
        #     l2_regularization_strength=0.1)
        # return linear_optimizer

        linear_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=float(FLAGS.linear_learning_rate)
        )
        return linear_optimizer

    def build_dnn_optimizer(self):
        dnn_optimizer = tf.train.AdagradOptimizer(
            learning_rate=float(FLAGS.dnn_learning_rate),
            initial_accumulator_value=0.1)
        return dnn_optimizer

    def build_estimator(self):
        linear_optimizer = self.build_linear_optimizer()
        dnn_optimizer = self.build_dnn_optimizer()
        dnn_hidden_units = [int(i) for i in FLAGS.dnn_hidden_units.split("-")]
        (linear_feature_columns, dnn_feature_columns) = self.build_feature_columns()

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


class DataInputPipeline(object):
    def __init__(self, input_files, batch_size=1000, num_epochs=20, feature_map_file="./feature_map.json"):
        self.batch_size = batch_size
        self.input_files = input_files
        self.epochs = num_epochs

        with open(feature_map_file, "r") as f:
            self.feature_info = json.load(f)

        input_file_list = []
        for input_file in self.input_files:
            if len(input_file) > 0:
                input_file_list.append(tf.train.match_filenames_once(input_file))
        self.filename_queue = tf.train.string_input_producer(
            tf.concat(input_file_list, axis=0),
            num_epochs=self.epochs,
            shuffle=True,
            capacity=512)
        self.reader = tf.TextLineReader(skip_header_lines=1)

        (self.column_dict, self.column_defaults) = self.build_column_format()

    def build_column_format(self):
        column_list = self.feature_info["columns"]
        column_dict = {field: index for index, field in enumerate(column_list)}

        column_defaults = [['']] * len(column_dict)
        for col in column_list:
            column_defaults[column_dict[col]] = [0.0]

        return column_dict, column_defaults

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
        label = tf.cast(samples[self.column_dict["clk"]], dtype=tf.float32)
        feature_dict_concat = {}

        for key, value in self.column_dict.items():
            if key == "clk" or value < 0 or value >= len(samples):
                continue
            if key in self.feature_info["selected_features"]:
                feature_dict_concat[key] = tf.cast(samples[value], dtype=tf.float32)

        return feature_dict_concat, label


def input_pipeline_test():
    logging.info("train data: %s" % (FLAGS.train_data))
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = DataInputPipeline(
        train_input_files,
        batch_size=1,
        feature_map_file=FLAGS.feature_map_file)  # FLAGS.batch_size)
    feature_batch, label_batch = train_input_pipeline.get_next_batch()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                print('*********')
                features, labels = sess.run([feature_batch, label_batch])
                print(features, labels)
        except tf.errors.OutOfRangeError:
            print("done!now lets kill all the threads...")
        finally:
            coord.request_stop()
            print('all threads are asked to stop!')

        coord.request_stop()
        coord.join(threads)

        # import pdb
        # pdb.set_trace()


def train_input_fn():
    train_input_files = FLAGS.train_data.strip().split(',')
    train_input_pipeline = DataInputPipeline(
        train_input_files,
        batch_size=FLAGS.batch_size,
        feature_map_file=FLAGS.feature_map_file)
    return train_input_pipeline.get_next_batch()


def train_model():
    if FLAGS.cold_start and tf.io.gfile.exists(FLAGS.model_dir):
        # tf.gfile.DeleteRecursively(FLAGS.model_dir)
        pass

    timeline_hook = tf.train.ProfilerHook(save_steps=1000, output_dir=os.path.join(
        os.getcwd(), './timeline_track'
    ))

    model = LogisticModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size,
        run_config=None,
        feature_map_file=FLAGS.feature_map_file)
    estimator = model.build_estimator()
    profile_hook = [timeline_hook]
    estimator.train(
        input_fn=lambda: train_input_fn(),
        steps=100,
        hooks=profile_hook)


def eval_input_fn():
    eval_input_files = FLAGS.eval_data.strip().split(',')
    eval_input_pipeline = DataInputPipeline(
        eval_input_files,
        batch_size=FLAGS.batch_size,
        feature_map_file=FLAGS.feature_map_file)
    return eval_input_pipeline.get_next_batch()


def eval_model():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % (FLAGS.model_dir))
        sys.exit(1)

    # Get the checkpoint path for evaluation
    checkpoint_path = None  # The latest checkpoint in model dir

    state = tf.train.get_checkpoint_state(
        checkpoint_dir=FLAGS.model_dir,
        latest_filename="checkpoint")
    if state and state.all_model_checkpoint_paths:
        if FLAGS.eval_ckpt_id == 0:
            checkpoint_path = state.all_model_checkpoint_paths[-1]
        else:
            checkpoint_path_sp = FLAGS.model_dir + "/model.ckpt-" + str(FLAGS.eval_ckpt_id)
            if checkpoint_path_sp in state.all_model_checkpoint_paths:
                checkpoint_path = checkpoint_path_sp
            else:
                logging.error("not find checkpoint id %d in %s" % (FLAGS.eval_ckpt_id, FLAGS.model_dir))
                checkpoint_path = None

    reader = tf.train.NewCheckpointReader(checkpoint_path)

    param_dict = reader.debug_string().decode("utf-8")

    # for key, val in param_dict.items():
    #     print(key, val)
    print('打印模型参数：参数名 数据类型 SHAPE')
    pprint.pprint(param_dict)
    # pdb.set_trace()
    # t1 = reader.get_tensor("linear/linear_model/age_level_0/weights")
    # pprint.pprint(t1)

    print("eval checkpoint path: %s" % checkpoint_path)
    eval_name = '' if checkpoint_path is None else str(FLAGS.eval_ckpt_id)

    model = LogisticModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size,
        run_config=None,
        feature_map_file=FLAGS.feature_map_file)
    estimator = model.build_estimator()
    eval_result = estimator.evaluate(
        input_fn=lambda: eval_input_fn(),
        steps=100,
        checkpoint_path=checkpoint_path,
        name=eval_name)
    print('eval_result', eval_result)


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


def export_savedmodel():
    if not tf.gfile.Exists(FLAGS.model_dir):
        logging.error("not find model dir: %s" % FLAGS.model_dir)
        sys.exit(1)

    model = LogisticModel(
        model_type=FLAGS.model_type,
        model_dir=FLAGS.model_dir,
        feature_size=FLAGS.feature_size,
        run_config=None,
        feature_map_file=FLAGS.feature_map_file)
    estimator = model.build_estimator()

    if FLAGS.savedmodel_mode == "raw":
        features = model.build_feature_dict()
        export_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(
            features=features,
            default_batch_size=None)
    elif FLAGS.savedmodel_mode == "parsing":
        (linear_feature_columns, dnn_feature_columns) = model.build_feature_columns()
        feature_columns = linear_feature_columns + dnn_feature_columns
        feature_spec = tf.feature_column.make_parse_example_spec(feature_columns)
        export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec=feature_spec,
            default_batch_size=None)
    else:
        logging.error("unsupported savedmodel mode: %s" % FLAGS.savedmodel_mode)
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

    model = LogisticModel(
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


def main():
    local_run()


# def test():
#     input_pipeline_test()


if __name__ == "__main__":
    main()
    # test()
