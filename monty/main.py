#!/usr/bin/env python3
from monty.input import CorruptedInputFunction
from monty.model import *

tf.enable_eager_execution()

if __name__ == '__main__':
    estimator = AutoEncoder(learning_rate=0.0001, model_dir="out", num_latent_variables=100)
    train_input_fn = CorruptedInputFunction()
    estimator.train(train_input_fn)
