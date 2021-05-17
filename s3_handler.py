from typing import Iterable, Optional, Union
import boto3
import pathlib
import os

_s3 = boto3.resource('s3')
_temp_path = pathlib.Path('./tmp')
if not _temp_path.exists():
    os.makedirs(_temp_path)

def download_from_s3(bucket : str, path : str, destination_path : Optional[os.PathLike] = None) -> os.PathLike:
    filename = path.split('/')[-1]
    destination_path = destination_path or _temp_path.joinpath(filename)
    
    _s3.Bucket(bucket).download_file(path, destination_path.as_posix())
    return destination_path

def upload_to_s3(bucket : str, source_path : os.PathLike, filename : str):
    _s3.Bucket(bucket).upload_file(source_path, filename)

def delete_from_s3(bucket : str, path : Union[str, Iterable[str]]):
    if isinstance(path, str):
        path = (path,)
    return _s3.Bucket(bucket).delete_objects(Delete={
        'Objects' : list([
            {'Key' : key } for key in path
        ])
    })