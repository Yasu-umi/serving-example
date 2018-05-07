# -*- coding: utf-8 -*-
import os
import time
import random

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import SGD


BATCH_SIZE = 100
EPOCHS = 1
X_FEATURE = 'x'
OUTPUTS = 'outputs'
L1 = 512
L2 = 256
LERANING_RATE = 0.1
MODEL_DIR = 'tmp/example2_keras/'

os.environ['PYTHONHASHSEED'] = '0'
tf.set_random_seed(42)
np.random.seed(42)
random.seed(42)

session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1
)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def create_keras_model(input_shape, n_classes):
    input = Input(shape=input_shape)
    l1 = Dense(L1, activation='relu')(input)
    l2 = Dense(L2, activation='relu')(l1)
    output = Dense(n_classes, activation='softmax')(l2)
    model = Model(inputs=[input], outputs=[output])
    return model


def save_keras_model(model):
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={X_FEATURE: tf.saved_model.utils.build_tensor_info(model.inputs[0])},
        outputs={OUTPUTS: tf.saved_model.utils.build_tensor_info(model.outputs[0])},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )
    export_dir = os.path.join(MODEL_DIR, str(int(time.time())))
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=export_dir)
    with K.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature},
        )
        builder.save()
    return export_dir


def main():
    K.set_learning_phase(0)

    mnist = input_data.read_data_sets('tmp/MNIST_data/', one_hot=True)

    model = create_keras_model(input_shape=(mnist.train.images.shape[1],), n_classes=mnist.train.labels.shape[1])
    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=LERANING_RATE),
        metrics=['accuracy'],
    )

    print('Training...')
    history = model.fit(
        x=mnist.train.images,
        y=mnist.train.labels.astype(np.int32),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=0,
        shuffle=False,
    )

    print('Done.\nEvaluating...')
    loss, accuracy = model.evaluate(
        x=mnist.test.images,
        y=mnist.test.labels.astype(np.int32),
        batch_size=mnist.test.images.shape[0],
        verbose=0,
    )
    print('loss: {}, accuracy: {}'.format(loss, accuracy))

    print('Saving Model...')
    export_dir= save_keras_model(model=model)
    print('Saved to {}'.format(export_dir))



if __name__ == '__main__':
    main()
