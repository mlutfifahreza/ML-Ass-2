import os
import cv2 as cv
from matplotlib import pyplot as plt
from mlp import *

datasets_path = 'downloads/flowers-datasets/'

# Import datasets
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
        img = cv.imread(datasets_path+target+'/'+item)
        # convert to grayscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # resize image
        img = cv.resize(img, (320,240))
        images[target].append(img)

# print("Sample image (2 each)")
# for target in targets:
#     print(target)
#     for i in range(2):
#         print("image",i)
#         print(images[target][i])
#     print()
#     input("enter to continue")

aClass = mlp("Fahreza",21)
aClass.myfunc()