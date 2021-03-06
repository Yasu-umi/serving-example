# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from grpc.beta import implementations
from tensorflow.examples.tutorials.mnist import input_data

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


X_FEATURE = 'x'
OUTPUTS = 'outputs'
SERVING_HOST = 'localhost'
SERVING_PORT = 8500


def main():
    mnist = input_data.read_data_sets('tmp/MNIST_data/', one_hot=True)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'example2_estimator'
    request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    feature = {
        X_FEATURE: tf.train.Feature(float_list=tf.train.FloatList(value=mnist.test.images[0])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    serialized = example.SerializeToString()
    examples = tf.make_tensor_proto(values=serialized, shape=[1])
    request.inputs['examples'].CopyFrom(examples)

    channel = implementations.insecure_channel(SERVING_HOST, SERVING_PORT)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
    stub.Predict.future(request, 1)
    result_future = stub.Predict.future(request, 1)
    result = result_future.result()
    print('answer: {}, prediction: {}'.format(np.argmax(mnist.test.labels[0]), np.argmax(result.outputs[OUTPUTS].float_val)))


if __name__ == '__main__':
    main()
