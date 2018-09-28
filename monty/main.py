#!/usr/bin/env python3
import tensorflow as tf

from monty import data


def encoder(x):
    x = tf.layers.dense(x, 100)
    return x


def decoder(x):
    x = tf.layers.dense(x, 2700, activation=lambda logits: tf.nn.sigmoid(logits) * 2)
    return x


if __name__ == '__main__':
    dataset_file_path = data.download_if_not_present('resources/PBMC.csv')
    dataset = data.create_dataset(dataset_file_path, 2700, 10, True, 1000)
    iterator = dataset.batch(batch_size=128, drop_remainder=True).make_one_shot_iterator()

    input = iterator.get_next()

    ae_out = decoder(encoder(input))
    loss = tf.reduce_mean(tf.square(ae_out - input))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    sess = tf.Session()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    with sess.as_default():
        while True:
            _, loss_value = sess.run([train_op, loss])
            print(loss_value)
