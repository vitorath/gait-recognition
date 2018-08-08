import os
from zipfile import ZipFile

DATA_PATH = 'data/raw/dataset.zip'
DATA_EXTRACT_TO = 'data/raw'

if (os.path.isfile(DATA_PATH)):
    print('Extract Dataset..')

    if not (os.path.isdir(DATA_EXTRACT_TO)):
        os.makedirs(DATA_EXTRACT_TO)

    zipReference = ZipFile(DATA_PATH, mode='r')
    zipReference.extractall(DATA_EXTRACT_TO)
    zipReference.close()
    print('Complete..')
else:
    print('Path not found!')