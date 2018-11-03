import numpy as np
import tensorflow as tf


def mock_input_fn():
    sequence = np.array([[[1, 1]]])

    def generator():
        for el in sequence:
            yield el

    dataset = tf.data.Dataset().batch(1).from_generator(generator,
                                                        output_types=tf.float64,
                                                        output_shapes=(tf.TensorShape([None, 2])))

    dataset = dataset.repeat(10000)
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()
    return el, el