import logging
import os

import s3fs
import tensorflow as tf

logger = logging.getLogger(__name__)


def download_if_not_present(local_file_path='resources/PBMC.csv'):
    s3bucket = 'pbmcasinglecell/'
    key = 'PBMC.csv'

    if not os.path.isfile(local_file_path):
        logging.info('downloading dataset')
        s3 = s3fs.S3FileSystem(anon=True)
        s3.get(s3bucket + key, local_file_path)

    logger.info('done downloading dataset')
    return local_file_path


def create_dataset(path_to_raw_dataset, num_epochs=10, batch_size=128, shuffle_buffer_size=1000, row_count=2700):
    filenames = [path_to_raw_dataset]
    record_defaults = [tf.int64] * row_count
    dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True,
                                         select_cols=[x for x in range(1, row_count + 1)])
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()



