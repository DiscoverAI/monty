import tensorflow as tf
from monty import FLAGS


def get_size_along_first_dim(tensor):
    first_dimension_size = tensor.shape.as_list()[0]
    return first_dimension_size if first_dimension_size is not None else FLAGS.batch_size


def set_index_list_to_zero(index_list, input_data):
    return tf.convert_to_tensor(tf.scatter_nd_update(tf.contrib.eager.Variable(input_data),
                                                     index_list,
                                                     tf.zeros(shape=tf.shape(index_list)[0])))
