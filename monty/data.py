import logging
import os

import s3fs

logger = logging.getLogger(__name__)


def download_if_not_present():
    s3bucket = 'pbmcasinglecell/'
    key = 'PBMC.csv'
    local_file_path = '../resources/PBMC.csv'

    if not os.path.isfile(local_file_path):
        logging.info('downloading dataset')
        s3 = s3fs.S3FileSystem(anon=True)
        s3.get(s3bucket + key, local_file_path)

    logger.info('done downloading dataset')
    return local_file_path
