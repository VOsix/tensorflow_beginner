import tensorflow as tf

features = {'color': [['R','A'], ['A','G'], ['A','G'], ['G','B'],['B','R']],
            'weight': [[1.0,2.0], [0.1,4.0], [5.0,1.0], [8.0,7.0],[3.0,2.0]]}

color_feature = tf.feature_column.categorical_column_with_hash_bucket(
                key = "color",
                hash_bucket_size = 10,
                dtype=tf.string)

column = tf.feature_column.weighted_categorical_column(color_feature, 'weight',dtype = tf.float32)

indicator = tf.feature_column.embedding_column(
    categorical_column = column,
    dimension = 5,
    combiner = "sqrtn")

tensor = tf.feature_column.input_layer(features, [indicator])

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    print(session.run([tensor]))