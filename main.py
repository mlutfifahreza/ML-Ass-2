import os
import cv2 as cv
from matplotlib import pyplot as plt
from mlp import *

datasets_path = 'downloads/flowers-datasets/'

# (1) ARSITEKTUR MLP 
# total input = 320*240 = 76800
width = 320
height = 240
dim = (width,height)
# total neuron di hidden (1 layer)
n_hidden = 240

# (3) LOAD DATASETS
targets = os.listdir(datasets_path)
if '.DS_Store' in targets: #remove clutter from mac folder file
    targets.remove('.DS_Store')
print('targets :',targets)

images = {}
for target in targets:
    images[target] = []
    image_names = os.listdir(datasets_path+target)
    if '.DS_Store' in image_names: #remove clutter from mac folder file
        image_names.remove('.DS_Store')
    for item in image_names:
        # get the image
        img = cv.imread(datasets_path+target+'/'+item)        
# (4) GRAYSCALE
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # resize image
        img = cv.resize(img, dim)
        # add each image to corresponding target in images
        images[target].append(img)

# (3) VISUALISASI DATA
# print("Sample image (2 each)")
# for target in targets:
#     print(target)
#     for i in range(2):
#         print("image",i)
#         print(images[target][i])
#     print()
#     input("enter to continue")

# (2) DEFINISIKAN ARSITEKTUR
flower_mlp = mlp(dim, n_hidden, targets)

# (12) SPLIT TRAIN AND TEST DATA

# (13) VISUALISASI : Error dan akurasi per epoch dg a=0.1 dan a=0.8