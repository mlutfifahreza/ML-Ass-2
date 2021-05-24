import cv2 as cv
from matplotlib import pyplot as plt
from mlp import *
import os

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

# (5) VISUALISASI DATA
# print("Sample image (2 each)")
# for target in targets:
#     print(target)
#     for i in range(1):
#         print("image",i)
#         # cv.imshow('image',images[target][i])
#         plt.imshow(images[target][i],cmap="gray")
#         plt.show()
#     print()
#     # input("enter to continue")
# print(images)
# image = {
#     "dandelion" : []
#     "rose"
#     "sunflower"
# }
# image["rose"][4]

fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 1
columns = 3

fig.add_subplot(rows, columns, 1)
# showing image
plt.imshow(images["dandelion"][0])
plt.axis('off')
plt.title("Dandelion")
  
# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(images["rose"][0])
plt.axis('off')
plt.title("Rose")
  
# Adds a subplot at the 3rd position
fig.add_subplot(rows, columns, 3)
  
# showing image
plt.imshow(images["sunflower"][0])
plt.axis('off')
plt.title("Sunflower")


# #important library to show the image 
# # import matplotlib.image as mpimg
# # import matplotlib.pyplot as plt
# #importing numpy to work with large set of data.
# import numpy as np
# #image read function
# img=cv.imread('Downloads/flowers-datasets/rose353897245_5453f35a8e.jpg')
# #image sclicing into 2D. 
# x=img[:,0]
# # x co-ordinate denotation. 
# plt.xlabel("Value")
# # y co-ordinate denotation.
# plt.ylabel("pixels Frequency")
# # title of an image .
# plt.title("Original Image")
# # imshow function with comperision of gray level value.
# plt.imshow(x,cmap="gray")
# #plot the image on a plane.
# plt.show()
# plt.title("HIstogramm for given Image'  ")
# plt.xlabel("Value")
# plt.ylabel("pixels Frequency")
# #hist function is used to plot the histogram of an image.
# plt.hist(x)


# (2) DEFINISIKAN ARSITEKTUR
# flower_mlp = mlp(dim, n_hidden, targets)

# (12) SPLIT TRAIN AND TEST DATA

# (13) VISUALISASI : Error dan akurasi per epoch dg a=0.1 dan a=0.8