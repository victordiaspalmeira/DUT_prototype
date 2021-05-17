import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s3_handler import *
import pathlib
import pytest

def get_file():
    file_to_upload = pathlib.Path('./tmp/test.txt')
    if not file_to_upload.exists():
        with open(file_to_upload, 'w+') as f: #create file
            f.write('Arquivo De Teste')
    return file_to_upload

def test_upload_delete_to_s3():
    bucket = 'intel-model-bucket'
    file_to_upload = get_file()
    upload_to_s3(bucket,file_to_upload.as_posix(), 'test') #test it doesn't raise
    delete_from_s3(bucket, 'test')

