'''
Created on Oct 29, 2018

@author: hauckjp
'''

import keras.backend as K
from keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, Reshape
from keras.models import Model
from keras.optimizers import SGD
import numpy as np
from cars import AnchorBoxFormat, LabelBoxInterpreter
import tensorflow as tf
from cars import tools

NUM_ANCHORS = 8
WEIGHT_REGIONS = 10
TRAINING_IMAGES = 60
TEST_IMAGES = 4
MINI_BATCHES_PER_IMAGE = 20
MINI_BATCH_SIZE = 32
TOTAL_ANCHOR_LOCATIONS = 8 * 8 * NUM_ANCHORS
EPOCHS = 5
FANCY_PROGRESS = True

imp = LabelBoxInterpreter("C:/Users/hauckjp/Downloads/testRexImport.json")
anchors = AnchorBoxFormat(tiles = 8, anchorPrototypes=[np.array([0, 0, 32/256, 32/256]), np.array([0, 0, 48/256, 32/256]), 
                                                       np.array([0, 0, 32/256, 48/256]), np.array([0, 0, 48/256, 48/256]), 
                                                       np.array([0, 0, 48/256, 64/256]), np.array([0, 0, 64/256, 48/256]),
                                                       np.array([0, 0, 32/256, 64/256]), np.array([0, 0, 64/256, 32/256])])

print('Downloading images...')
imp.downloadImages(range(TRAINING_IMAGES + TEST_IMAGES), displayProgress = FANCY_PROGRESS)

print("Generating model...")
K.set_session(tools.sess)
trainingImages = tf.image.convert_image_dtype(imp.getTensorFlowImages(range(TRAINING_IMAGES), displayProgress = False), tf.float32)
trainingImages.eval(session = tools.sess)

testImages = tf.image.convert_image_dtype(imp.getTensorFlowImages(range(TRAINING_IMAGES, TEST_IMAGES + TRAINING_IMAGES), displayProgress = False), tf.float32)
testImages.eval(session = tools.sess)

miniBatches = []
for j in range(MINI_BATCHES_PER_IMAGE):
    miniBatches.append(imp.getMiniBatches(range(TRAINING_IMAGES), anchors, MINI_BATCH_SIZE))

labelTraining = tf.Variable(np.zeros((1, 8, 8, NUM_ANCHORS, 2)), dtype = tf.float32)
blankLabel = -tf.ones((1, 8, 8, NUM_ANCHORS, 2), dtype = tf.float32)
foregroundTraining = tf.Variable(np.zeros((1, 8, 8, NUM_ANCHORS, 4)), dtype = tf.float32)
blankForeground = -100 * tf.ones((1, 8, 8, NUM_ANCHORS, 4), dtype = tf.float32)
tools.sess.run(labelTraining.initializer)
tools.sess.run(foregroundTraining.initializer)
labelTraining.eval(session=tools.sess)
blankLabel.eval(session=tools.sess)
foregroundTraining.eval(session=tools.sess)
blankForeground.eval(session=tools.sess)

def getConv2D_BN(inp, filters, kernel_shape, activation='relu'):
    x = Conv2D(filters, kernel_shape, padding = 'same')(inp)
    x = BatchNormalization()(x)
    if not (activation is None):
        x = Activation(activation)(x)
    return x

inputImages = Input(shape=(256,256,3), batch_shape=(1, 256, 256, 3), name = 'input', dtype = np.float32)

# something like VGG-16
x = getConv2D_BN(inputImages, 64, (3,3))    # 256x256x64 = 4194304
x = getConv2D_BN(x, 64, (3,3))              # 256x256x64 = 4194304
x = MaxPool2D(pool_size = (2,2))(x)         # 128x128x64 = 1048576
x = getConv2D_BN(x, 128, (3,3))             # 128x128x128 = 2097152
x = getConv2D_BN(x, 128, (3,3))             # 128x128x128 = 2097152
x = MaxPool2D(pool_size = (2,2))(x)         # 64x64x128 = 524288
x = getConv2D_BN(x, 256, (3,3))             # 64x64x256 = 1048576
x = getConv2D_BN(x, 256, (3,3))             # 64x64x256 = 1048576
x = getConv2D_BN(x, 256, (3,3))             # 64x64x256 = 1048576
x = MaxPool2D(pool_size = (2,2))(x)         # 32x32x256 = 262144
x = getConv2D_BN(x, 512, (3,3))             # 32x32x512 = 524288
x = getConv2D_BN(x, 512, (3,3))             # 32x32x512 = 524288
x = getConv2D_BN(x, 512, (3,3))             # 32x32x512 = 524288
x = MaxPool2D(pool_size = (2,2))(x)         # 16x16x512 = 131072
x = getConv2D_BN(x, 512, (3,3))             # 16x16x512 = 131072
x = getConv2D_BN(x, 512, (3,3))             # 16x16x512 = 131072
x = getConv2D_BN(x, 512, (3,3))             # 16x16x512 = 131072
x = MaxPool2D(pool_size = (2,2))(x)         # 8x8x512 = 32768 -- might end up stopping with the 16x16, we'll see

# now for the interesting part: RPN = Region Proposal Network
convolved = getConv2D_BN(x, 512, (3,3))     # yes, there are more convolutions 8x8x512 = 32768
# sigmoid for labels since output should be in [0,1] (can't use softmax because classes aren't disjoint!), no activation for regions
proposedLabels = getConv2D_BN(convolved, 2 * NUM_ANCHORS, (1,1), activation = 'sigmoid')    # use a 1x1 kernel convolution to avoid having to use a FC layer
proposedRegions = getConv2D_BN(convolved, 4 * NUM_ANCHORS, (1,1), activation = None)

# now the proposedLabels output has shape (8x8x{2*NUM_ANCHORS}), so if op is the output, we would get whether an object
# occurs in tile (x,y) with bounding box similar to anchor i by by op[x][y][2 * i] == 1 <=> object appears <=> 
# op[x][y][2 * i + 1] == 0 (we have TWO outputs per tile. We have two outputs per anchor:
# first is p, the probability of an object, second is 1-p, probability of not object, i.e. it's a two-class classifier;
# however, the classes are "stacked" on top of one another. e.g. with 2 anchors, output for one tile = [.2, .8, .1, .9]
# = [p_1, 1 - p_1, p_2, 1 - p_2], p_1 probability of object similar to anchor 1, p_2 " anchor2

# proposedRegions output has shape (8x8x{4*NUM_ANCHORS}), so if op is the output tensor, we would get the proposed region
# for anchor with center in tile (x,y) and index i by op[x][y][4 * i: 4 * (i + 1)] = [dx_i, dy_i, dw_i, dh_i], where
# dx_i, etc. are deviations of predicted box from corresponding anchor box. RCNN also recommends some fancy normalization 
# on the coordinates that needs to be undone to get actual pixel-coordinate predictions:
# t_x = (x - x_a) / w_a
# t_y = (y - y_a) / h_a
# t_w = log(w/w_a)
# t_h = log(h/h_a)
# where t_* is the training value used, *_a is the corresponding anchor value, x,y,w,h are the ground truth values,
# so each training data box is normalized with respect to its corresponding anchor
# we want that our model outputs d*_i = t_*_i

# now we need (well, would prefer) the labels to have shape (8, 8, NUM_ANCHORS, 2) and
# proposal regions to have shape (8, 8, NUM_ANCHORS, 4)
shapedLabels = Reshape(target_shape=(8, 8, NUM_ANCHORS, 2))(proposedLabels)
shapedRegions = Reshape(target_shape=(8, 8, NUM_ANCHORS, 4))(proposedRegions)

# now we can finally build the model
model = Model(inputs=inputImages, outputs=[shapedLabels, shapedRegions])

# to compile we need to define the custom loss functions, for which we need the current minibatch
# this is why we need to train the model with a batch size of 1. we need the loss function to vary per minibatch

def binaryCrossEntropy(y_true, y_pred):
    return -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

# binary cross-entropy for labels (but only on the anchor outputs that are in the minibatch)
def lossLabels(y_true, y_pred):
    return K.sum(binaryCrossEntropy(y_true, y_pred) * K.cast(K.not_equal(y_true, blankLabel), dtype=tf.float32))

# smooth L1 for regions
HUBER_DELTA = 0.5
def smoothL1(y_true, y_pred):
    x = K.abs(y_true - y_pred)
    x = K.switch(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
    return x

def lossRegions(y_true, y_pred):
    return K.sum(smoothL1(y_true, y_pred) * K.cast(K.not_equal(y_true, blankForeground), dtype=tf.float32))

optimizer = SGD(lr = .005, momentum = .9)
model.compile(optimizer = optimizer, metrics = ['accuracy'], loss=[lossLabels, lossRegions], loss_weights=[1, WEIGHT_REGIONS])

lastLoss = [0]

# now, at long last, to train the model
print("Training model... (if you get allocator warnings, don't worry about it...)")
for i in range(MINI_BATCHES_PER_IMAGE):
    if not FANCY_PROGRESS:
        print("Training minibatch", i)
    for j in range(TRAINING_IMAGES):
        if not FANCY_PROGRESS:
            print("Training image", j)
        else:
            tools.printProgressBars((i * TRAINING_IMAGES + j, j), (MINI_BATCHES_PER_IMAGE * TRAINING_IMAGES, TRAINING_IMAGES),
                                    ("Training minibatches", "Training images in this minibatch"),
                                    ("of minibatches trained", "images trained. Max last loss: " + str(lastLoss[0])),
                                    lengths=(30, 15))
        tools.setMasks(labelTraining, foregroundTraining, -1, -100, miniBatches[i][j], 0)
        lastLoss = model.train_on_batch(trainingImages[j:j+1], [labelTraining, foregroundTraining])

if FANCY_PROGRESS:
    tools.printProgressBars((MINI_BATCHES_PER_IMAGE * TRAINING_IMAGES, TRAINING_IMAGES), (MINI_BATCHES_PER_IMAGE * TRAINING_IMAGES, TRAINING_IMAGES),
                                    ("Training minibatches", "Training images in this minibatch"),
                                    ("of minibatches trained", "images trained. Max last loss: " + str(lastLoss[0])),
                                    lengths=(30, 15))
prediction = [model.predict(testImages[i:(i + 1)], steps = 1) for i in range(TEST_IMAGES)]
