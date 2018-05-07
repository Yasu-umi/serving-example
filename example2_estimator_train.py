# -*- coding: utf-8 -*-
import os
import random

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 100
EPOCHS = 1
X_FEATURE = 'x'
OUTPUTS = 'outputs'
L1 = 512
L2 = 256
LERANING_RATE = 0.1
MODEL_DIR = 'tmp/example2_estimator/'

os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(42)
np.random.seed(42)
random.seed(42)


def model_fn(features, labels, mode, params):
    x_feature = features[X_FEATURE]
    l1 = tf.layers.dense(
        inputs=x_feature,
        units=L1,
        activation=tf.nn.relu,
    )
    l2 = tf.layers.dense(
        inputs=l1,
        units=L2,
        activation=tf.nn.relu,
    )
    logits = tf.layers.dense(
        inputs=l2,
        units=params['n_clssses'],
        activation=None,
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(input_tensor=losses)
        learning_rate = params['learning_rate']
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate,
        )
        train_op = optimizer.minimize(loss=loss)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits=logits)
        predictions_dict = {
            OUTPUTS: probabilities,
        }
        export_outputs = {
            OUTPUTS: tf.estimator.export.PredictOutput(outputs=predictions_dict),
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions_dict, export_outputs=export_outputs)
    elif mode == tf.estimator.ModeKeys.EVAL:
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(input_tensor=losses)
        predictions = tf.nn.softmax(logits=logits)
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    else:
        raise Exception('InvalidMode')


def save_estimator(estimator, x_shape, default_batch_size=1):
    feature_spec = {
        X_FEATURE: tf.FixedLenFeature(dtype=tf.float32, shape=[default_batch_size] + x_shape),
    }
    serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        feature_spec=feature_spec,
        default_batch_size=default_batch_size,
    )
    export_dir = estimator.export_savedmodel(
        export_dir_base=MODEL_DIR,
        serving_input_receiver_fn=serving_input_receiver_fn,
    ).decode('utf-8')
    return export_dir


def main():
    mnist = input_data.read_data_sets('tmp/MNIST_data/', one_hot=True)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.train.images},
        y=mnist.train.labels.astype(np.int32),
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=False,
    )
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={X_FEATURE: mnist.test.images},
        y=mnist.test.labels.astype(np.int32),
        batch_size=mnist.test.images.shape[0],
        num_epochs=1,
        shuffle=False,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=MODEL_DIR,
        params={'learning_rate': LERANING_RATE, 'n_clssses': mnist.train.labels.shape[1]},
    )

    print('Training...')
    estimator.train(input_fn=train_input_fn)

    print('Done.\nEvaluating...')
    result = estimator.evaluate(input_fn=test_input_fn)
    print('loss: {}, accuracy: {}'.format(result['loss'], result['accuracy']))

    print('Saving Model...')
    export_dir = save_estimator(estimator=estimator, x_shape=list(mnist.train.images.shape[1:]))
    print('Saved to {}'.format(export_dir))


if __name__ == '__main__':
    main()
