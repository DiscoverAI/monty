import logging
import os

import tensorflow as tf

logging.basicConfig(
    format="%(asctime)s %(levelname)s	[%(process)d] %(module)s %(filename)s %(funcName)s %(message)s",
    level=os.environ.get("LOGLEVEL", "DEBUG")
)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 128,
                            """Number of data items to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 10,
                            """Number of learning epochs.""")
tf.app.flags.DEFINE_float('learning_rate', 0.0001, """Learning rate.""")
tf.app.flags.DEFINE_integer('num_features', 2700,
                            """Number of features (types of RNA transcripts) in the data.""")
tf.app.flags.DEFINE_string('dataset_path', 'resources/PBMC.csv',
                           """Path to the dataset.""")
tf.app.flags.DEFINE_integer('minimum_library_size', 1,
                            """Minimum sum of expressed genes that a measurement needs to have,
                            in order to be processed""")
tf.app.flags.DEFINE_integer('minimum_expressed_genes', 1,
                            """Minimum count of nonzero gene expression that a measurement needs to have,
                            in order to be processed""")
tf.app.flags.DEFINE_integer('impute_iterations', 20,
                            """How many times the imputation should iterate at each step""")
tf.app.flags.DEFINE_integer('mask_percentage', 0,
                            """How many percent of nonzero measurements
                            should be masked during imputing autoencoder training""")
tf.app.flags.DEFINE_string('model_dir', 'out',
                           """Directory for model persistence""")