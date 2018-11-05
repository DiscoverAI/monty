import numpy as np
from monty.ops import *


def test_get_size_on_first_dim():
    assert get_size_along_first_dim(tf.convert_to_tensor([1, 2])) == 2


def test_assign_zero_at_index_list():
    out = set_index_list_to_zero(tf.constant([[1, 1], [1, 2]], dtype=tf.int32, shape=[2, 2]),
                                 tf.constant([[5, 6, 7], [8, 9, 10]], dtype=tf.float32, shape=[2, 3]))
    assert np.allclose(out.numpy(), [[5, 6, 7], [8, 0, 0]])
