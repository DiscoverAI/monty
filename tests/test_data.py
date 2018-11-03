import os

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from monty import data

tf.enable_eager_execution()

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
    iterator = data.create_dataset("test_resources/PBMC_test.csv", 5, 1, False, None).batch(2).make_one_shot_iterator()
    npt.assert_array_equal(iterator.get_next(), np.array([[2, 1, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=np.float32))


def test_drop_outliers():
    dataset = data.create_dataset("test_resources/PBMC_test.csv", 5, 1, False, None)
    non_outliers = data.drop_outliers(dataset, minimum_expressed_genes=1, minimum_library_size=100)
    iterator = non_outliers.batch(2).make_one_shot_iterator()
    first_batch = iterator.get_next()

    npt.assert_array_equal(first_batch, np.array([[41, 42, 43, 44, 45]], dtype=np.float32))
    assert first_batch.shape == (1, 5)


def test_normalize_data():
    dataset = data.create_dataset("test_resources/PBMC_test.csv", 5, 1, False, None) \
        .batch(1)
    dataset = data.normalize_dataset(dataset)
    iterator = dataset.make_one_shot_iterator()
    npt.assert_allclose(iterator.get_next()[0], np.vectorize(lambda x: np.log(x + 1))([2, 1, 0, 1, 0]))
