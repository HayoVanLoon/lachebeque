from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_hub as hub


tf.logging.set_verbosity(tf.logging.INFO)

TEXT_FEATURE = 'text'
SCORE_TYPE = -1

DEBUG = False

MAX_SEQ_LEN = -1

HIDDEN_LAYER_SIZE = -1
CELL_SIZE = -1


def debug_print(x):
    if DEBUG:
        print(x)


def init(hparams):
    global MAX_SEQ_LEN, HIDDEN_LAYER_SIZE, CELL_SIZE, DEBUG, SCORE_TYPE
    DEBUG = hparams['debug']

    SCORE_TYPE = {'score': 2,
                  'norm_score': 3,
                  'norm_log_score': 4}[hparams['score_type']]

    MAX_SEQ_LEN = hparams['max_sequence_length']

    HIDDEN_LAYER_SIZE = hparams['hidden_layer_size']
    CELL_SIZE = hparams['cell_size']


def embed_words_from_text(texts):
    embed = hub.Module("https://tfhub.dev/google/nnlm-en-dim50-with-normalization/1")

    cleaned = tf.regex_replace(texts, '[[:punct:]]', ' ')
    debug_print('>>> cleaned: {}'.format(cleaned))

    words = tf.strings.split(cleaned)
    debug_print('>>> words: {}'.format(words))

    densed = tf.sparse.to_dense(words, default_value='')
    debug_print('>>> densed: {}'.format(densed))

    paddings = [[0, 0], [0, MAX_SEQ_LEN - tf.shape(densed)[1]]]
    padded = tf.pad(densed, paddings, 'CONSTANT', constant_values='')
    debug_print('>>> padded: {}'.format(padded))

    reshaped = tf.reshape(padded, [-1])
    debug_print('>>> reshaped: {}'.format(reshaped))

    embedded = embed(reshaped)
    dim = embedded.shape[1]

    return embedded, dim


def lachebeque1_model(texts, mode, params):
    embedded, dim = embed_words_from_text(texts)
    debug_print('>>> embedded: {}'.format(embedded))

    reshaped = tf.reshape(embedded, [-1, MAX_SEQ_LEN * dim, 1])
    debug_print('>>> reshaped: {}'.format(reshaped))

    cell = tf.nn.rnn_cell.GRUCell(CELL_SIZE)
    output, state = tf.nn.dynamic_rnn(cell, reshaped, dtype=tf.float32)

    h1 = tf.layers.dense(state, HIDDEN_LAYER_SIZE, activation=tf.nn.relu)

    prediction = tf.layers.dense(h1, 1, activation=None)

    return prediction


def lachebeque2_model(texts, mode, params):
    embedded, dim = embed_words_from_text(texts)
    debug_print('>>> embedded: {}'.format(embedded))

    reshaped = tf.reshape(embedded, [-1, MAX_SEQ_LEN, dim, 1])
    debug_print('>>> reshaped: {}'.format(reshaped))

    c1 = tf.layers.conv2d(reshaped, filters=4, kernel_size=5, padding='same')
    debug_print('>>> c1: {}'.format(c1))
    p1 = tf.layers.max_pooling2d(c1, pool_size=3, strides=2)
    debug_print('>>> p1: {}'.format(p1))

    c2 = tf.layers.conv2d(p1, filters=8, kernel_size=5, padding='same')
    debug_print('>>> c2: {}'.format(c2))
    p2 = tf.layers.max_pooling2d(c2, pool_size=3, strides=2)
    debug_print('>>> p2: {}'.format(p2))

    flattened = tf.reshape(p2, [-1, p2.shape[1] * p2.shape[2] * p2.shape[3], 1])
    debug_print('>>> flattened: {}'.format(flattened))

    cell = tf.nn.rnn_cell.GRUCell(CELL_SIZE)
    output, state = tf.nn.dynamic_rnn(cell, flattened, dtype=tf.float32)

    h1 = tf.layers.dense(state, HIDDEN_LAYER_SIZE, activation=tf.nn.relu)

    prediction = tf.layers.dense(h1, 1, activation=None)

    return prediction


def get_input_fn(file, mode=True, batch_size=256):

    def input_fn():

        def decode_csv(xs):
            columns = tf.decode_csv(xs,
                                    record_defaults=[[''], [0.0]],
                                    select_cols=[1, SCORE_TYPE])
            texts = columns[0]
            label = columns[1]
            return texts, label

        dataset = tf.data.TextLineDataset(file).map(decode_csv)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size=10 * batch_size)
        else:
            num_epochs = 1

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn


def serving_input_fn():
    text_placeholder = tf.placeholder(tf.string, [None])
    feature_placeholders = {TEXT_FEATURE: text_placeholder}
    return tf.estimator.export.TensorServingInputReceiver(text_placeholder,
                                                          feature_placeholders)


def joke_regressor(features, labels, mode, params):
    model_functions = {
        'lachebeque1': lachebeque1_model,
        'lachebeque2': lachebeque2_model
    }
    model_function = model_functions[params['model']]

    predictions = model_function(features, mode, params)[:, -1]

    debug_print('>>> mode: {}'.format(mode))
    debug_print('>>> predictions: {}'.format(predictions))
    debug_print('>>> labels: {}'.format(labels))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(
            labels=labels,
            predictions=predictions
        )
        eval_metric_ops = {
            'rmse': tf.metrics.mean_squared_error(labels=labels,
                                                  predictions=predictions)
        }
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                loss=loss, global_step=tf.train.get_global_step())
        else:
            train_op = None
    else:
        loss = None
        train_op = None
        eval_metric_ops = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs={'predictions': tf.estimator.export.RegressionOutput(predictions)}
    )


def train_and_evaluate(output_dir, hparams):
    get_train = get_input_fn(file=hparams['train_data_path'],
                             mode=tf.estimator.ModeKeys.TRAIN,
                             batch_size=hparams['train_batch_size'])
    get_eval = get_input_fn(file=hparams['eval_data_path'],
                            mode=tf.estimator.ModeKeys.EVAL,
                            batch_size=512)

    run_config = tf.estimator.RunConfig(save_checkpoints_secs=hparams['min_eval_frequency'])

    estimator = tf.estimator.Estimator(model_fn=joke_regressor,
                                       params=hparams,
                                       config=run_config,
                                       model_dir=output_dir)
    train_spec = tf.estimator.TrainSpec(input_fn=get_train,
                                        max_steps=hparams['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=get_eval,
                                      steps=None,
                                      exporters=exporter,
                                      start_delay_secs=hparams['eval_delay_secs'],
                                      throttle_secs=hparams['min_eval_frequency'])
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
