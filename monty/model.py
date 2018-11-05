import monty.data as data
import tensorflow as tf
from monty import FLAGS


def encoder(x, training, num_latent_variables):
    x = tf.layers.dense(x, 200, activation=tf.nn.relu)
    x = tf.layers.batch_normalization(x, training=True)
    x = tf.layers.dense(x, num_latent_variables, activation=tf.nn.relu)
    return x


def decoder(x, training):
    x = tf.layers.dense(x, 200, activation=tf.nn.relu)
    x = tf.layers.batch_normalization(x, training=True)
    x = tf.layers.dense(x, FLAGS.num_features, activation=tf.nn.relu)
    return x


def autoencoder(inputs, num_latent_variables, mode):
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    net = encoder(inputs, is_training, num_latent_variables)
    net = decoder(net, is_training)
    return net


def _create_estimator_spec(labels, logits, learning_rate, mode):
    predictions = {"prediction": tf.round(data.denormalize_op(logits))}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions)

    train_op = None
    eval_metric_ops = None
    loss = tf.losses.mean_squared_error(labels, logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = \
            tf.train.AdamOptimizer(beta1=0.5, learning_rate=learning_rate) \
                .minimize(loss, global_step=tf.train.get_global_step())

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            "rmse": tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float64), tf.cast(logits, tf.float64))
        }

    logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=5)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=[logging_hook],
        eval_metric_ops=eval_metric_ops)


class AutoEncoder(tf.estimator.Estimator):
    def __init__(self, learning_rate, model_dir, num_latent_variables, config=None):
        def _model_fn(features, labels, mode):
            logits = autoencoder(inputs=features, mode=mode, num_latent_variables=num_latent_variables)
            return _create_estimator_spec(
                labels=labels,
                logits=logits,
                learning_rate=learning_rate,
                mode=mode
            )

        super(AutoEncoder, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config
        )
