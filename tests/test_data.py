import os

import tensorflow as tf
from monty import data

test_file_path = 'resources/PBMC_test.csv'


def test_download_file_not_present(mocker):
    try:
        os.remove(test_file_path)
    except OSError:
        pass
    mocked_s3 = mocker.patch('s3fs.S3FileSystem.get')

    assert data.download_if_not_present(local_file_path=test_file_path) == test_file_path
    mocked_s3.assert_called_once_with('pbmcasinglecell/PBMC.csv', test_file_path)


def test_download_file_present(mocker):
    try:
        os.remove(test_file_path)
    except OSError:
        pass
    dataset = open(test_file_path, 'w')
    dataset.write(' ')
    mocked_s3 = mocker.patch('s3fs.S3FileSystem.get')

    assert data.download_if_not_present(local_file_path=test_file_path) == test_file_path
    assert mocked_s3.call_count == 0


def test_create_dataset():
    iterator = data.create_dataset("test_resources/PBMC_test.csv", batch_size=4, shuffle_buffer_size=8)
    sess = tf.Session()
    with sess.as_default():
        tf.train.start_queue_runners(sess)
        first_batch = sess.run(iterator)
        print(first_batch)
        assert len(first_batch) == 2700
