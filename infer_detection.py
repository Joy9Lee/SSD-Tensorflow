import tensorflow as tf
import itertools
import time
import numpy as np

from nets import nets_factory
from preprocessing import preprocessing_factory

# =========================================================================== #
# Some default EVAL parameters
# =========================================================================== #
# List of recalls values at which precision is evaluated.
LIST_RECALLS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
                0.90, 0.95, 0.96, 0.97, 0.98, 0.99]
DATA_FORMAT = 'NHWC'

# =========================================================================== #
# SSD evaluation Flags.
# =========================================================================== #
tf.app.flags.DEFINE_float(
    'select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer(
    'select_top_k', 400, 'Select top-k detected bounding boxes.')
tf.app.flags.DEFINE_integer(
    'keep_top_k', 200, 'Keep top-k detected objects.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Non-Maximum Selection threshold.')
tf.app.flags.DEFINE_float(
    'matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer(
    'eval_resize', 4, 'Image resizing: None / CENTRAL_CROP / PAD_AND_RESIZE / WARP_RESIZE.')
tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size.')

# =========================================================================== #
# Main evaluation flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_string(
    'inference_graph_path', '',
    'The path where the frozen graph was written.')
tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_string(
    'input_tfrecord_paths', None, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')


FLAGS = tf.app.flags.FLAGS


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        with tf.Session() as sess:
            # =========================================================================== #
            # Main evaluation flags.
            # =========================================================================== #

            # Get the SSD network and its anchors.
            ssd_class = nets_factory.get_network(FLAGS.model_name)
            ssd_params = ssd_class.default_params._replace(num_classes=FLAGS.num_classes)
            ssd_net = ssd_class(ssd_params)

            # Evaluation shape and associated anchors: eval_image_size
            ssd_shape = ssd_net.params.img_shape

            # Select the preprocessing function.
            preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
            image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                preprocessing_name, is_training=False)

            # =================================================================== #
            # Create a dataset provider.
            # =================================================================== #

            input_tfrecord_paths = [
                v for v in FLAGS.input_tfrecord_paths.split(',') if v]
            filename_queue = tf.train.string_input_producer(
                input_tfrecord_paths, shuffle=False, num_epochs=1)

            tf_record_reader = tf.TFRecordReader()
            _, serialized_example_tensor = tf_record_reader.read(filename_queue)

            feature_map = {
                "image/encoded":
                    tf.FixedLenFeature([], tf.string)
            }
            features = tf.parse_single_example(serialized_example_tensor, feature_map)

            encoded_image = features["image/encoded"]
            image_tensor = tf.image.decode_image(encoded_image, channels=3)
            image_tensor.set_shape([None, None, 3])

            image_tensor, _1, _2, _3 = \
                image_preprocessing_fn(image_tensor, labels=None, bboxes=None,
                                       out_shape=ssd_shape,
                                       data_format=DATA_FORMAT,
                                       resize=FLAGS.eval_resize,
                                       difficults=None)

            image_tensor = tf.placeholder(tf.float32, shape=[1, 300, 300, 3], name="Placeholder")
            image_data = np.ones([1, 300, 300, 3])

            # =================================================================== #
            # Import SSD Network
            # =================================================================== #
            with tf.gfile.FastGFile(FLAGS.inference_graph_path, 'rb') as graph_def_file:
                graph_content = graph_def_file.read()
                graph_def = tf.GraphDef()
                graph_def.MergeFromString(graph_content)

                tf.import_graph_def(
                    graph_def,
                    input_map={'eval_batch:0': tf.to_float(image_tensor)},
                    name="",
                )

                graph = tf.get_default_graph()

                predictions = [
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/softmax/Reshape_1:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/softmax_1/Reshape_1:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/softmax_2/Reshape_1:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/softmax_3/Reshape_1:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/softmax_4/Reshape_1:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/softmax_5/Reshape_1:0')]

                localisations = [
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/block4_box/Reshape:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/block7_box/Reshape:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/block8_box/Reshape:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/block9_box/Reshape:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/block10_box/Reshape:0'),
                    graph.get_tensor_by_name(
                        'ssd_300_vgg/block11_box/Reshape:0')]

            # =================================================================== #
            # Inference loop.
            # =================================================================== #
            sess.run(tf.local_variables_initializer())
            tf.train.start_queue_runners()
            start = time.time()

            try:
                for counter in range(0, 100):
                    tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                                           counter)
                    _pre, _loc = sess.run([predictions, localisations], feed_dict={'Placeholder:0': image_data})
            except tf.errors.OutOfRangeError:
                tf.logging.info('Finished processing records')

            # Log time spent.
            elapsed = time.time()
            elapsed = elapsed - start
            print('Time spent : %.3f seconds.' % elapsed)
            print('Time spent per BATCH: %.3f seconds.' % (elapsed / 100))


if __name__ == '__main__':
    tf.app.run()

