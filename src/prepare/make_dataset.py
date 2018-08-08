import os
import shutil
from tqdm import tqdm

DATA_PATH = 'data/raw/dataset/'
SAVE_TO = 'data/full/dataset/'

WALK_TYPE = ["nm-01", "nm-02", "nm-03", "nm-04", "nm-05", "nm-06"]
WALK_PERSPECTIVE = ["000","072", "090", "108"]
SAMPLES = ['001', '003', '004', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '017', '018', '019', '020', 
           '021', '025', '026', '027', '029', '030', '032', '035', '038', '039', '041', '042', '043', '047','049', '050', '052', 
           '056', '057', '059', '060', '061', '062', '063', '066', '069', '071', '072', '073', '078', '080', '082', '085', '086', 
           '087', '088', '089', '090', '091', '092', '093', '097', '099', '100', '102', '103', '105', '106', '107', '108', '110',
           '111', '113', '114', '115', '117', '118', '119', '120', '121', '122', '123', '124']

if (os.path.exists(SAVE_TO)):
    os.makedirs(SAVE_TO)

# dataset/sample/walk-type/walk-perspective
for samples in tqdm(SAMPLES):
    for walkType in WALK_TYPE:
        for walkPerspective in WALK_PERSPECTIVE:
            tmpPath = samples + '/' + walkType + '/' + walkPerspective
            shutil.copytree(DATA_PATH + tmpPath, SAVE_TO + tmpPath)

