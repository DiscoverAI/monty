from monty.model import *
import os
import shutil
import numpy as np


def test_encoder_model():
    batch_size = 2
    FLAGS.num_features = 5
    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 5])
    y = encoder(x, training=True, num_latent_variables=2)
    assert y.shape == (batch_size, 2)


def test_decoder_model():
    batch_size = 2
    FLAGS.num_features = 5
    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])
    y = decoder(x, training=True)
    assert y.shape == (batch_size, 5)


def test_autoencoder_model():
    batch_size = 2
    FLAGS.num_features = 5
    x = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])
    y = autoencoder(x, num_latent_variables=100, mode=tf.estimator.ModeKeys.TRAIN)
    assert y.shape == (batch_size, 5)


def mock_input_fn():
    sequence = np.array([[[1, 1]]])

    def generator():
        for el in sequence:
            yield el

    dataset = tf.data.Dataset().batch(1).from_generator(generator,
                                                        output_types=tf.float64,
                                                        output_shapes=(tf.TensorShape([None, 2])))
    dataset = dataset.repeat(500)
    iter = dataset.make_one_shot_iterator()
    el = iter.get_next()
    return el, el


def test_estimator():
    """Tests if our model can learn the identity mapping of a constant dataset in 1000 steps"""
    if not os.path.exists("tests/out"):
        os.mkdir("tests/out")
    FLAGS.num_features = 2
    estimator = AutoEncoder(0.1, "tests/out", num_latent_variables=2)
    assert estimator.model_dir == "tests/out"
    estimator.train(input_fn=mock_input_fn, steps=500)
    prediction = estimator.predict(input_fn=mock_input_fn).__next__()
    assert prediction["prediction"].shape == (2,)
    assert np.array_equal(prediction["prediction"], [1, 1])
    shutil.rmtree("tests/out")
