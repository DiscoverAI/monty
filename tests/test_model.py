import os
import shutil

from monty.model import *
from tests.mocks import *


def test_encoder_model():
    batch_size = 2
    FLAGS.num_features = 5
    x = tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=tf.float32, shape=[batch_size, 5])
    y = encoder(x, training=True, num_latent_variables=2)
    assert y.shape == (batch_size, 2)


def test_decoder_model():
    batch_size = 2
    FLAGS.num_features = 5
    x = tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=tf.float32, shape=[batch_size, 5])
    y = decoder(x, training=True)
    assert y.shape == (batch_size, 5)


def test_autoencoder_model():
    batch_size = 2
    FLAGS.num_features = 5
    x = tf.constant([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], dtype=tf.float32, shape=[batch_size, 5])
    y = autoencoder(x, num_latent_variables=100, mode=tf.estimator.ModeKeys.TRAIN)
    assert y.shape == (batch_size, 5)


def test_estimator():
    """Tests if our model can learn the identity mapping of a constant dataset in 1000 steps"""
    if not os.path.exists("tests/out"):
        os.mkdir("tests/out")
    FLAGS.num_features = 2
    estimator = AutoEncoder(0.1, "tests/out", num_latent_variables=2)
    assert estimator.model_dir == "tests/out"
    estimator.train(input_fn=mock_input_fn, steps=1000)
    prediction = estimator.predict(input_fn=mock_input_fn).__next__()
    assert prediction["prediction"].shape == (2,)
    assert np.array_equal(prediction["prediction"], np.round(data.denormalize_op([1.0, 1.0])))
    shutil.rmtree("tests/out")
