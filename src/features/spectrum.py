import cv2
import numpy as np 
from math import ceil

def regionOfInterest(image):
    """
    param: 
        image -> matriz

    Por meio da técnica de 'Bounding Box' é extraida a região de interesse.
    -> Bounding Box: é uma técnica no qual é calculodo os extremos de uma imagem para limitar o processamento
    """
    offset = 2
    nonZeroImage = np.nonzero(image)
    maxWidth, maxHeight = np.amax(nonZeroImage[1]), np.amax(nonZeroImage[0])
    minWidth, minHeight = np.amin(nonZeroImage[1]), np.amin(nonZeroImage[0])
    return image[minHeight-offset : maxHeight+offset, minWidth-offset : maxWidth+offset].copy()

def centralizer(image, shape, width=120, height=240):
    """
    params:
        image -> matriz
        shape -> lista 2 dimensões
        width -> largura maxima
        height -> altura maxima

    return:
        imagem (matriz) centralizada

    Centralizer a imagem tendo como base o retangulo (não o centroid).
    Além disso, a imagem é redimensionada e centralizar (sem alterar as dimensões da area de interesse)
    """
    base = np.zeros((height, width), np.uint8)
    centerHeight = ceil( height/2 - shape[0]/2 )
    centerWidth = ceil( width/2 - shape[1]/2 )
    base[centerHeight : centerHeight + shape[0], centerWidth : centerWidth + shape[1]] = image.copy();
    return base.copy()

def spectrum(imagesSequence, width=120, height=240):
    """
    params:
        imagesSequencia -> lista de matrizes
        width -> largura maxima
        height -> altura maxima

    return:
        imagem (matriz) do movimento

    Cria um spectro do movimento conforme a sequencia de imagens
    """
    spectrum = np.zeros((height,width), np.uint8)
    # Primeira imagem para referencia
    oldImage = regionOfInterest(imagesSequence[0])
    oldImage = centralizer(oldImage, oldImage.shape)
    
    for image in imagesSequence[1:]:
        currImage = regionOfInterest(image)
        currImage = centralizer(currImage, currImage.shape)
        diffImages = cv2.bitwise_xor(oldImage, currImage)
        spectrum = np.sum([spectrum, diffImages], axis=0)
        oldImage = currImage.copy()

    # Normalizar
    #spectrum = np.divide(spectrum, len(imagesSequence))
    spectrum = np.divide(spectrum, np.max(spectrum))
    return spectrum

"""
DATA_PATH = 'data/full/dataset/'
img = cv2.imread(DATA_PATH + '006/nm-03/108/050.png',0)

img = regionOfInterest(img)

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""