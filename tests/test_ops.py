import tensorflow as tf
from monty.ops import *


def test_get_size_on_first_dim():
    assert get_size_along_first_dim(tf.convert_to_tensor([1, 2])) == 2
