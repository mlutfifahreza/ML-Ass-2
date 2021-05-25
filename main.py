import cv2 as cv
import random
from matplotlib import pyplot as plt
import os
from mlp import *

datasets_path = 'downloads/flowers-datasets/'

# (1) ARSITEKTUR MLP 
# total input = 320*240 = 76800
width,height = 120,80
# width,height = 140,100
# height = 240
dim = (width,height)
n_hidden = 40 # total neuron di hidden (1 layer)

# (3) LOAD DATASETS
print("import database")
targets = os.listdir(datasets_path)
if '.DS_Store' in targets: #remove clutter from mac folder file
    targets.remove('.DS_Store')
print('targets :',targets)

encoding = {
    "dandelion" : [1,0,0],
    "rose" : [0,1,0],
    "sunflower" : [0,0,1]
}

print("getting images")
images = []
for target in targets:
    image_names = os.listdir(datasets_path+target)
    if '.DS_Store' in image_names: #remove clutter from mac folder file
        image_names.remove('.DS_Store')
    for item in image_names:
        img = cv.imread(datasets_path+target+'/'+item)  # get the image
# (4) GRAYSCALE
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # resize image
        img = cv.resize(img, dim)
        # add each image,encoding to images
        images.append([img,encoding[target]])


# (12) SPLIT TRAIN AND TEST DATA
import random
print("split data")
def split_data(data):
    n_shuffle = 4
    n_data = len(data)
    for i in range(n_shuffle):
        random.shuffle(data)
    boundary = int(0.8 * n_data)
    return data[0:boundary], data[boundary:n_data]

train_data, test_data = split_data(images)
print("total train data =",len(train_data))
print("total test data =",len(test_data))

# (2) DEFINISIKAN ARSITEKTUR
print("new mlp")
flower_mlp = mlp(dim, n_hidden, targets, 20, 0.5)
print("new mlp training")
flower_mlp.train(train_data, test_data)
print(flower_mlp.train_log)

# # (5) VISUALISASI DATA


# # (13) VISUALISASI : Error dan akurasi per epoch dg a=0.1 dan a=0.8