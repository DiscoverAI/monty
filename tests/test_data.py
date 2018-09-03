import os

from monty import data


def test_download_file_not_present(mocker):
    try:
        os.remove('resources/PBMC.csv')
    except OSError:
        pass
    mocked_s3 = mocker.patch('s3fs.S3FileSystem.get')

    assert data.download_if_not_present() == 'resources/PBMC.csv'
    mocked_s3.assert_called_once_with('pbmcasinglecell/PBMC.csv', 'resources/PBMC.csv')


def test_download_file_present(mocker):
    try:
        os.remove('resources/PBMC.csv')
    except OSError:
        pass
    dataset = open('resources/PBMC.csv', 'w')
    dataset.write(' ')
    mocked_s3 = mocker.patch('s3fs.S3FileSystem.get')

    assert data.download_if_not_present() == 'resources/PBMC.csv'
    assert mocked_s3.call_count == 0
