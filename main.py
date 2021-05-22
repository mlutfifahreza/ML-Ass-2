import os
import cv2 as cv
from matplotlib import pyplot as plt

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
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        images[target].append(gray)

# try printing 3 images of each target
plt.subplot(331),plt.imshow(images[targets[0]][0], 'gray'),plt.title(targets[0])
plt.subplot(332),plt.imshow(images[targets[0]][1], 'gray'),plt.title(targets[0])
plt.subplot(333),plt.imshow(images[targets[0]][2], 'gray'),plt.title(targets[0])

plt.subplot(334),plt.imshow(images[targets[1]][0], 'gray'),plt.title(targets[1])
plt.subplot(335),plt.imshow(images[targets[1]][1], 'gray'),plt.title(targets[1])
plt.subplot(336),plt.imshow(images[targets[1]][2], 'gray'),plt.title(targets[1])

plt.subplot(337),plt.imshow(images[targets[2]][0], 'gray'),plt.title(targets[2])
plt.subplot(338),plt.imshow(images[targets[2]][1], 'gray'),plt.title(targets[2])
plt.subplot(339),plt.imshow(images[targets[2]][2], 'gray'),plt.title(targets[2])
plt.show()