from decoder import *
import numpy as np

import time
from tqdm import tqdm
import cv2
import os
# import matplotlib.pyplot as plt

for file_name in os.listdir('src'):
    t = time.time()
    file_path = 'src/'+file_name
    print(f'starting to process file {file_path}')
    # decode the xml file and store the information in 'result', a list of 'spectrum' objects
    result = decode(file_path)

    print('initializing min and max...')
    min_mz = min([np.amin(spectrum.mzArr) for spectrum in result])
    max_mz = max([np.amax(spectrum.mzArr) for spectrum in result])
    min_rt = min([spectrum.retentionTime for spectrum in result])
    max_rt = max([spectrum.retentionTime for spectrum in result])

    # setting width and height, this can be any number we set
    HEIGHT = 1*len(result)
    WIDTH = int(max([spectrum.mzArr[len(spectrum.mzArr)-1] - spectrum.mzArr[0] for spectrum in result])+1)
    WIDTH *= 5
    HEIGHT *= 1

    print('converting spectra list to 2d array...')
    # initialize img as a 2d array specified height and width
    img = np.zeros((HEIGHT, WIDTH))
    # iterate through the list of spectra
    for spectrum in tqdm(result):
        rt = spectrum.retentionTime
        # the y_index of all the peaks in this spectrum
        y_index = int((rt - min_rt)/(max_rt - min_rt)*(HEIGHT-1)+0.5)
        for mz, intensity in zip(spectrum.mzArr, spectrum.intenArr):
            # the x_index of the current peak
            x_index = int((mz-min_mz)/(max_mz-min_mz)*(WIDTH-1)+0.5)
            img[y_index][x_index] += intensity

    min_intensity = min([np.amin(spectrum) for spectrum in img])
    max_intensity = max([np.amax(spectrum) for spectrum in img])
    max_intensity /= 10

    print('normalizing intensities...')
    # eheheheheheheheeheheh
    for i in tqdm(range(len(img))):
        for j in range(len(img[i])):
            img[i, j] = int((img[i, j] - min_intensity)/(max_intensity - min_intensity)*255)
            img[i, j] = min(img[i, j], 255)

    file_name = file_name.split('.')[0]
    np.save('misc/'+file_name, img)
    np.save('misc/'+file_name+'_misc', np.array([max_rt, min_rt, max_mz, min_mz]))
    print("image status:" + str(cv2.imwrite('img1' + file_name + ".png", img)))
