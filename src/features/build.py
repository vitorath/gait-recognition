import os
import sys
import cv2
import numpy as np 
from tqdm import tqdm

sys.path.append('src/features/spectrum')
from spectrum import spectrum 

DATA_PATH = 'data/full/dataset/'
SAVE_TO = 'data/full/spectrum-feature.npy'

spectrumDataset = []
# Rename labels for peaples classification
label = 1
for sample in tqdm(os.listdir(DATA_PATH)):
    for walkType in os.listdir(DATA_PATH + sample):
        for walkPerspective in os.listdir(DATA_PATH + sample + '/' +  walkType):
            imagesSequence = []
            for imageName in os.listdir(DATA_PATH + sample + '/' +  walkType + '/' + walkPerspective):
                if imageName[-4:] == '.png':
                    imagePath = DATA_PATH + sample + '/' +  walkType + '/' + walkPerspective + '/' + imageName
                    image = cv2.imread(imagePath, 0)
                    if (np.size(np.nonzero(image)) > 0):
                        imagesSequence.append(image)
            
            img = spectrum(imagesSequence)
            spectrumDataset.append([np.array(img).astype(np.float32), np.array(label)])
            
    label += 1

np.save(SAVE_TO, spectrumDataset)
