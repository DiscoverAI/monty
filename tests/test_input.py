from monty.input import *
import tensorflow as tf
import numpy as np


def test_input_function():
    batch_size = 3
    FLAGS.dataset_path = "fake_path"
    noise_input_fn = UnchangedInputFunction(tf.estimator.ModeKeys.TRAIN,
                                            dataset_path="test_resources/PBMC_test.csv",
                                            batch_size=batch_size,
                                            num_epochs=1,
                                            minimum_expressed_genes=0,
                                            minimum_library_size=0,
                                            num_features=5,
                                            shuffle=False)
    x, y = noise_input_fn()
    sess = tf.Session()
    expected = data.normalize_op(
        tf.constant(
            [[2, 1, 0, 1, 0],
             [0, 0, 0, 0, 0],
             [41, 42, 43, 44, 45]]
            , dtype=tf.float32))
    x, y, expected = sess.run([x, y, expected])
    assert x.shape == (batch_size, 5)
    assert y.shape == (batch_size, 5)
    assert np.allclose(y, expected, atol=0.01)
