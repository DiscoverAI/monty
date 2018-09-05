import os

import tensorflow as tf
from monty import data

TEST_FILE_PATH = 'resources/PBMC_test.csv'


def test_download_file_not_present(mocker):
    try:
        os.remove(TEST_FILE_PATH)
    except OSError:
        pass
    mocked_s3 = mocker.patch('s3fs.S3FileSystem.get')

    assert data.download_if_not_present(local_file_path=TEST_FILE_PATH) == TEST_FILE_PATH
    mocked_s3.assert_called_once_with('pbmcasinglecell/PBMC.csv', TEST_FILE_PATH)


def test_download_file_present(mocker):
    try:
        os.remove(TEST_FILE_PATH)
    except OSError:
        pass
    dataset = open(TEST_FILE_PATH, 'w')
    dataset.write(' ')
    mocked_s3 = mocker.patch('s3fs.S3FileSystem.get')

    assert data.download_if_not_present(local_file_path=TEST_FILE_PATH) == TEST_FILE_PATH
    assert mocked_s3.call_count == 0
    os.remove(TEST_FILE_PATH)


def test_create_dataset():
    iterator = data.create_dataset(
        "test_resources/PBMC_test.csv",
        num_epochs=1,
        shuffle=False,
        batch_size=4,
        shuffle_buffer_size=None)
    sess = tf.Session()
    with sess.as_default():
        first_batch = sess.run(iterator)
        assert first_batch.shape == (4, 2700)
        assert first_batch[0][0] == 2
        assert first_batch[0][1] == 1
        assert first_batch[1][0] == 0
        assert first_batch[1][1] == 1
