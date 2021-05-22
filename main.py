import cv2 as cv
import os

datasets_path = 'downloads/flowers-datasets/'

# Import datasets
targets = os.listdir(datasets_path)
if '.DS_Store' in targets: #remove clutter from mac folder file
    targets.remove('.DS_Store')
print('targets :',targets)

images = {}
for target in targets:
    img = os.listdir(datasets_path+target)
    images[target] = img
print(images)