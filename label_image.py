from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
                                input_mean=0, input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.io.read_file(file_name, input_name)  # Update tf.read_file to tf.io.read_file
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(file_reader, channels=3, name='png_reader')
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(tf.image.decode_gif(file_reader, name='gif_reader'))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
    else:
        image_reader = tf.image.decode_jpeg(file_reader, channels=3, name='jpeg_reader')
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

    # Do not create a session here
    return normalized

def load_labels(label_file):
    label = []

    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()  # Update tf.gfile.GFile to tf.io.gfile.GFile
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())

    return label

def main(img):
    file_name = img
    model_file = "retrained_graph.pb"
    label_file = "retrained_labels.txt"
    input_height = 224
    input_width = 224
    input_mean = 128
    input_std = 128
    input_layer = "input"
    output_layer = "final_result"

    graph = load_graph(model_file)
    t = read_tensor_from_image_file(file_name,
                                    input_height=input_height,
                                    input_width=input_width,
                                    input_mean=input_mean,
                                    input_std=input_std)

    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:  # Create a session within the 'with' block
        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t.numpy()})



    results = np.squeeze(results)
    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    for i in top_k:
        return labels[i]

if __name__ == "__main__":
    img = "path_to_image.jpg"  # Replace with the path to your image file
    emotion_label = main(img)
    print("Emotion: ", emotion_label)
