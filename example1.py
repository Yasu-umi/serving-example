import os
import time
import tensorflow as tf
from tensorflow.core.framework import types_pb2
from grpc.beta import implementations
from google.protobuf import wrappers_pb2

from tensorflow_serving.apis import prediction_service_pb2
from tensorflow_serving.apis import predict_pb2

if __name__ == '__main__':
    tf.app.flags.DEFINE_integer('version', 0, 'version')
    tf.app.flags.DEFINE_integer('x', 0, 'x')
    tf.app.flags.DEFINE_integer('y', 0, 'y')

    MODEL_NAME = 'default'
    VERSION = tf.app.flags.FLAGS.version
    SERVING_HOST = 'localhost'
    SERVING_PORT = 9000
    X = tf.app.flags.FLAGS.x
    Y = tf.app.flags.FLAGS.y
    EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'tmp', str(VERSION))

    # define graph
    graph = tf.Graph()
    with graph.as_default():
      x = tf.placeholder(dtype=tf.int64, shape=(), name='x')
      y = tf.placeholder(dtype=tf.int64, shape=(), name='y')
      x_add_y = tf.add(x=x, y=y)

    # test run graph
    with tf.Session(graph=graph) as sess:
        print('local x_add_y run result: {}'.format(sess.run(x_add_y, feed_dict={x: X, y: Y})))

    # save current sess graph for serving
    builder = tf.saved_model.builder.SavedModelBuilder(EXPORT_DIR)
    with tf.Session(graph=graph) as sess:
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'x_add_y': tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'x': tf.saved_model.utils.build_tensor_info(x), 'y': tf.saved_model.utils.build_tensor_info(y)},
                    outputs={'x_add_y':  tf.saved_model.utils.build_tensor_info(x_add_y)},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
                ),
            },
        )
        builder.save()

    # wait serving load
    time.sleep(1)

    # create grpc stub
    channel = implementations.insecure_channel(SERVING_HOST, SERVING_PORT)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # create predict request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    version = wrappers_pb2.Int64Value()
    version.value = VERSION
    request.model_spec.version.CopyFrom(version)
    request.model_spec.signature_name = 'x_add_y'

    request.inputs['x'].dtype = types_pb2.DT_INT64
    request.inputs['x'].int64_val.append(X)
    request.inputs['y'].dtype = types_pb2.DT_INT64
    request.inputs['y'].int64_val.append(Y)

    result_future = stub.Predict.future(request, 1)
    result = result_future.result()
    print('serving x_add_y run result: {}'.format(result.outputs['x_add_y'].int64_val[0]))
