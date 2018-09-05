#!/usr/bin/env python3
from monty import data
import tensorflow as tf


def encoder(x):
    x = tf.layers.dense(x, 100)
    return x


def decoder(x):
    x = tf.layers.dense(x, 2700, activation=lambda logits: tf.nn.sigmoid(logits) * 2)
    return x


if __name__ == '__main__':
    dataset_file_path = data.download_if_not_present('resources/PBMC.csv')
    inputs = data.create_dataset(dataset_file_path,
                                 num_epochs=10,
                                 shuffle=True,
                                 batch_size=128,
                                 shuffle_buffer_size=1000)

    ae_out = decoder(encoder(inputs))
    loss = tf.reduce_mean(tf.square(ae_out - inputs))
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

    sess = tf.Session()
    tf.train.start_queue_runners(sess)
    sess.run(tf.global_variables_initializer())
    batch = sess.run(inputs)
    with sess.as_default():
        while True:
            _, loss_value = sess.run([train_op, loss])
            print(loss_value)
