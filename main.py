from decoder import *
import numpy as np

import time
from tqdm import tqdm
import cv2
# import matplotlib.pyplot as plt

# file_name = 'resources/130124_dilA_9_04.mzML'
file_name = 'resources/newMS1.mzML'
#file_name = 'resources/newMS2.mzML'
t = time.time()
print(f'starting to process file {file_name}')

# decode the xml file and store the information in 'result', a list of 'spectrum' objects
result = decode(file_name)

# setting width and height, this can be any number we set
WIDTH, HEIGHT = 1*len(result[0].mzArr), 1*len(result)
WIDTH = int(max([spectrum.mzArr[len(spectrum.mzArr)-1] - spectrum.mzArr[0] for spectrum in result])+1)
WIDTH *= 10
HEIGHT *= 1

print('initializing min and max...')
min_mz = min([np.amin(spectrum.mzArr) for spectrum in result])
max_mz = max([np.amax(spectrum.mzArr) for spectrum in result])
min_rt = min([spectrum.retentionTime for spectrum in result])
max_rt = max([spectrum.retentionTime for spectrum in result])

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

'''
print(f'normalizing intensity values...\ttime: {time.time()-t}')
# find the max and min of all the intensity values
mx, mn = -sys.maxsize, sys.maxsize
for e in tqdm(result):
    mx, mn = max(mx, max(e.intenArr)), min(mn, min(e.intenArr))
# by decreasing the max, the weak intensity values get assigned a larger number and is therefore scaled up in the image
mx /= 10
# adjust intensity values in range 0-255, the weakest gets assigned 0 and the strongest 255
for e in tqdm(result):
    for i in range(len(e.intenArr)):
        e.intenArr[i] = max(0, min(255, int(((e.intenArr[i]-mn)/(mx-mn))*255)))

print(f"normalizing m/z values...\ttime:{time.time()-t}")
# find the max and min of all the mz values
mx, mn = -sys.maxsize, sys.maxsize
for e in tqdm(result):
    mx, mn = max(mx, max(e.mzArr)), min(mn, min(e.mzArr))
# convert the mzArrays
for e in tqdm(result):
    e.normalize(WIDTH, mx, mn)

# normalize result based on retention time

# normalize retention time
low, high = sys.maxsize, -sys.maxsize
for element in tqdm(result):
    low = min(low, element.retentionTime)
    high = max(high, element.retentionTime)
new = np.zeros((HEIGHT, WIDTH))
for current in tqdm(result):
    index = int(((current.retentionTime - low) / (high-low)) * (len(new)-1))
    new[index] = np.add(current.normalized, new[index])
print('normalized retention time')
'''
np.save(file_name+'_npscan', img)
np.save(file_name+'_misc', np.array([max_rt, min_rt, max_mz, min_mz]))
print("image status:" + str(cv2.imwrite(file_name.split('.')[0] + ".png", img)))
