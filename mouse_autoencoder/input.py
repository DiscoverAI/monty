import boto3
import botocore
import os


def download_if_not_present():
    path_to_file = os.getcwd() + "/PBMC.csv"

    if os.path.isfile(path_to_file):
        return

    BUCKET_NAME = 'pbmcasingelcell'
    KEY = 'PBMC.csv'

    s3 = boto3.resource('s3')

    try:
        s3.Bucket(BUCKET_NAME).download_file(KEY, os.getcwd() + "/PBMC.csv")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise
