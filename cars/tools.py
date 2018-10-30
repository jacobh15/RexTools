'''
Created on Oct 22, 2018

You'll need to install Pillow and requests if you haven't already. They're used to
get the image from the url and then get its width and height (though in the future
you'll need to use the image itself as input) for normalization.

To install Pillow and requests you'll need pip, which you can install using the
python installer I think. Then in a command prompt navigate to 
<path to your python directory>\Python\Scripts and run
pip install Pillow
pip install requests

Put these modules somewhere your PYTHONPATH points to,
and you should be able to use them in python.

@author: hauckjp
'''

import cars.labelreader as keys
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import tensorflow as tf
import numpy as np
import random
import subprocess

KEY_X_CENTER = 'xc'
KEY_Y_CENTER = 'yc'
KEY_WIDTH = 'width'
KEY_HEIGHT = 'height'
KEY_IMAGE = 'image'
KEY_TF_IMAGE = 'tf_image'

sess = tf.Session()

# calculates the car vertices in normalized image coordinates
def normalizedGeometry(imageName, imageData):
    if KEY_IMAGE in imageData[imageName]:
        width = imageData[imageName][KEY_IMAGE][KEY_WIDTH]
        height = imageData[imageName][KEY_IMAGE][KEY_HEIGHT]
    else:
        i = getImage(imageName, imageData)  # get image at the url for imageName
        width = i.width
        height = i.height
    cars = imageData[imageName][keys.LOCAL_KEY_CARS] # get car data for imageName
    points = []
    for car in cars:    # normalize car points and add to points list
        points.append(normalized(pointDictToNPArray(car[keys.KEY_GEOM]), width, height))
    return points

def pointDictToNPArray(points):
    pointA = np.zeros((len(points), 2))
    for i, p in enumerate(points):
        pointA[i] = np.array((p['x'], p['y']))
    return pointA

# normalizes a list of points so that (0, 0) -> (0, 0)
# and (width, height) -> (1, 1)
def normalized(points, width, height):  
    pointsN = []
    for p in points:
        pointsN.append((p[0] / width, p[1] / height))
    return pointsN

# returns a Pillow image object for the given imageName
def getImage(imageName, imageData):
    url = imageData[imageName][keys.LOCAL_KEY_URL]
    return Image.open(BytesIO(requests.get(url).content))

# returns a tensorflow image object
def getTensorFlowImage(imageName, imageData):
    if KEY_IMAGE in imageData[imageName]:
        return imageData[imageName][KEY_IMAGE][KEY_TF_IMAGE]
    url = imageData[imageName][keys.LOCAL_KEY_URL]
    dataString = requests.get(url).content
    tfIm = tf.image.decode_png(dataString)
    tfIm.set_shape((256,256,3))
    return tfIm

# downloads the images with the given names and returns a tensorflow tensor containing their data
# also records the width and height and stores the subtensor of each image in imageData
def downloadImages(imageNames, imageData, eachTime = None):
    tfImages = getTensorFlowImages(imageNames, imageData, eachTime = eachTime)
    tfImagesResolved = tfImages.eval(session = sess)
    width = tfImagesResolved.shape[1]
    height = tfImagesResolved.shape[2]
    for index, name in enumerate(imageNames):
        imageData[name][KEY_IMAGE] = {KEY_TF_IMAGE: tfImages[index], KEY_WIDTH: width, KEY_HEIGHT: height}
    
    return tfImages

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', end = '\r'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = end)
    # Print New Line on Complete
    if iteration == total: 
        print()

def printProgressBars(iterations, totals, prefixes = None, suffixes = None, decimals = 1, lengths = 100, fill = '█'):
    if prefixes is None:
        prefixes = [''] * len(iterations)
    if suffixes is None:
        suffixes = [''] * len(iterations)
    percents = [''] * len(iterations)
    try:
        lengthit = [l for l in iter(lengths)]
    except TypeError:
        lengthit = [lengths] * len(iterations)
    allDone = True
    for i in range(len(iterations)):
        allDone = allDone and iterations[i] == totals[i]
        percents[i] = ("{0:." + str(decimals) + "f}").format(100 * (iterations[i] / float(totals[i])))
        filledLength = int(lengthit[i] * iterations[i] / totals[i])
        bar = fill * filledLength + "-" * (lengthit[i] - filledLength)
        print('%s |%s| %s%% %s' % (prefixes[i], bar, percents[i], suffixes[i]), end = '\n' if i < len(iterations) - 1 and not allDone else '\r')
    
    if not allDone:
        for i in range(len(iterations) - 1):
            subprocess.call('', shell=True)
            print('\033[F', end='')
    else:
        print()

# returns a tensor of images (a batch)
def getTensorFlowImages(imageNames, imageData, eachTime = None):
    images = []
    for index, name in enumerate(imageNames):
        images.append(getTensorFlowImage(name, imageData))
        if not (eachTime is None):
            eachTime(index)
    if not (eachTime is None):
        eachTime(len(imageNames))
    return tf.stack(images)

def drawBoundingBox(image, box, normalized = True):
    box2 = box[:]
    if normalized:
        box2[0] *= image.width
        box2[2] *= image.width
        box2[1] *= image.height
        box2[3] *= image.height
    
    draw = ImageDraw.Draw(image)
    minx = box2[0] - .5 * box2[2]
    maxx = box2[0] + .5 * box2[2]
    miny = box2[1] - .5 * box2[3]
    maxy = box2[1] + .5 * box2[3]
    draw.line((minx, miny, minx, maxy), fill = (0, 255, 0, 255))
    draw.line((minx, maxy, maxx, maxy), fill = (0, 255, 0, 255))
    draw.line((maxx, maxy, maxx, miny), fill = (0, 255, 0, 255))
    draw.line((maxx, miny, minx, miny), fill = (0, 255, 0, 255))

def clipBoxes(boxes, clipConf):
    s = boxes.shape
    keepers = []
    for tilex in range(0, s[0]):
        for tiley in range(0, s[1]):
            if boxes[tilex][tiley][0] > clipConf:
                b = boxes[tilex][tiley]
                keepers.append([b[1], b[2], b[3], b[4]])
    return keepers;

# non max suppression
def removeRepeats(boxes, iouLimit):
    removeIndices = [False] * len(boxes)
    for indexB, b in enumerate(boxes):
        for index, check in enumerate(boxes):
            if check != b:
                iou = intersectionOverUnion(b[1:], check[1:])
                if iou > iouLimit:
                    if check[0] > b[0]:
                        removeIndices[indexB] = True
                    else:
                        removeIndices[index] = True
    boxes2 = []
    for i in range(0, len(boxes)):
        if not removeIndices[i]:
            boxes2.append(boxes[i])
    return boxes2

# caluclates iou (duh)
def intersectionOverUnion(box1, box2):
    xMin = max(box1[0] - .5 * box1[2], box2[0] - .5 * box2[2])
    yMin = max(box1[1] - .5 * box1[3], box2[1] - .5 * box2[3])
    yMax = min(box1[0] + .5 * box1[2], box2[0] + .5 * box2[2])
    xMax = min(box1[1] + .5 * box1[3], box2[1] + .5 * box2[3])
    
    intersect = max(0, xMax - xMin) * max(0, yMax - yMin)
    
    box1Area = box1[2] * box1[3]
    box2Area = box2[2] * box2[3]
    
    union = box1Area + box2Area - intersect
    
    return intersect / union

def toTupleList(dictList):
    tl = []
    for d in dictList:
        tl.append((d['x'], d['y']))
    return tl

def boxDictToArray(boxDict):
    return np.array([boxDict['xc'], boxDict['yc'], boxDict['width'], boxDict['height']])
    
# calculates the bounding boxes of the cars in imageName
def boundingBoxes(imageName, imageData):
    myCars = imageData[imageName][keys.LOCAL_KEY_CARS]
    boxes = []
    for car in myCars:
        boxes.append(boundingBox(car[keys.KEY_GEOM]))
    return boxes

# calculates the bounding boxes in normalized coordinates (see normalize) for the cars in imageName
def normalizedBoundingBoxes(imageName, imageData):
    nGeom = normalizedGeometry(imageName, imageData)
    boxes = []
    for pointsN in nGeom:
        boxes.append(boundingBox(pointsN))
    
    return boxes

# calculates training output for the given imageNames and tiling size as a set of nested
# python lists (shape = (batch_size, tiles, tiles, 5)), batch_size = len(imageNames)
def getBatchTiledTrainingData(imageNames, imageData, tiles):
    training = []
    for name in imageNames:
        training.append(getTiledTrainingData(name, imageData, tiles))
    return training

def genPrototypes(sizes, width, height):
    prototypes = []
    scalediv = np.array([width, height])
    for size in sizes:
        prototypes.append(np.append(np.array([0, 0]), (size / scalediv)))
        prototypes.append(np.append(np.array([0, 0]), np.array([2 * size[0], size[1]]) / scalediv))
        prototypes.append(np.append(np.array([0, 0]), np.array([size[0], 2 * size[1]]) / scalediv))
    return prototypes

# tuple of arrays of tuples of (anchor, index = (tileX, tileY, i), boxIndex (if positive), classification (length 2 array: [fg, bg]))
# first array are positives, second negatives
# each anchor is assigned a positive label if
#    (i) it has the highest iou with a ground-truth box
#    (ii) it has an iou with a ground-truth box greater than foregroundCutoff
# each anchor is assigned a negative label if it has iou less than backgroundCutoff for all ground-truth boxes
# anchors that are neither positive nor negative are excluded
# positive label = 1, negative label = 0
def getLabeledAnchors(boxes, tiles, anchorPrototypes, foregroundCutoff, backgroundCutoff, width, height):
    k = len(anchorPrototypes)
    potentialAnchors = np.zeros((tiles, tiles, k, 5))
    scale = np.array([width, height, width, height])
    iouMaxes = [-float('inf')] * len(boxes)
    iouMaxIndices = [0] * len(boxes)
    for tileX in range(tiles):
        for tileY in range(tiles):
            for i in range(k):
                potentialAnchors[tileX][tileY][i] = createAnchor(tileX, tileY, i, tiles, anchorPrototypes, width, height)
                doExclude = True
                for j in range(len(boxes)):
                    iou = intersectionOverUnion(boxes[j] * scale, potentialAnchors[tileX][tileY][i])
                    if iou > foregroundCutoff:
                        potentialAnchors[tileX][tileY][i][4] = j
                        doExclude = False
                    elif iou > backgroundCutoff:
                        doExclude = False
                    if iou > iouMaxes[j]:
                        iouMaxes[j] = iou
                        iouMaxIndices[j] = (tileX, tileY, i)
                if doExclude:
                    potentialAnchors[tileX][tileY][i][4] = -1
    
    for j in range(len(boxes)):
        tileX, tileY, i = iouMaxIndices[j]
        potentialAnchors[tileX][tileY][i][4] = j
    
    positives = []
    negatives = []
    for tileX in range(tiles):
        for tileY in range(tiles):
            for i in range(k):
                if potentialAnchors[tileX][tileY][i][4] > -2:
                    if potentialAnchors[tileX][tileY][i][4] > -1:
                        positives.append((potentialAnchors[tileX][tileY][i][0:4], (tileX, tileY, i), potentialAnchors[tileX][tileY][i][4], [1, 0]))
                    else:
                        negatives.append((potentialAnchors[tileX][tileY][i][0:4], (tileX, tileY, i), np.array([0, 1])))
    return (positives, negatives)

def setMasks(labelTraining, foregroundTraining, blankLabelVal, blankForegroundVal, miniBatch, batchNumber):
    labelTrainingNP = np.full(labelTraining.shape, blankLabelVal, dtype=np.float32)
    foregroundTrainingNP = np.full(foregroundTraining.shape, blankForegroundVal, dtype=np.float32)
    for i in range(len(miniBatch[0])):
        tileX, tileY, a = miniBatch[0][i][0]
        labelTrainingNP[batchNumber][tileX][tileY][a] = np.array(miniBatch[0][i][1], dtype=np.float32)
    for i in range(len(miniBatch[1])):
        tileX, tileY, a = miniBatch[1][i][0]
        foregroundTrainingNP[batchNumber][tileX][tileY][a] = miniBatch[1][i][1]
    labelTraining = tf.convert_to_tensor(labelTrainingNP, dtype = tf.float32)
    foregroundTraining = tf.convert_to_tensor(foregroundTrainingNP, dtype = tf.float32)
    labelTraining.eval(session=sess)
    foregroundTraining.eval(session=sess)
    
def createAnchor(tileX, tileY, i, tiles, anchorPrototypes, width, height):
    scale = np.array([width, height])
    return np.append(np.append(np.array(anchorPrototypes[i][0:2]) * scale + np.array([tileX + .5, tileY + .5]) / tiles * scale, np.array(anchorPrototypes[i][2:4]) * scale), np.array([-2]))

# returns anchor training data as the difference in center and dimension between an anchor and the ground-truth    
# using anchor normalization (see testRPN)
def getAnchorDeviations(boxes, positiveAnchors, width, height):
    deviations = np.zeros((len(positiveAnchors), 4))
    scale = np.array([width, height, width, height])
    for i in range(len(positiveAnchors)):
        groundTruth = boxes[int(positiveAnchors[i][2])] * scale
        anchor = positiveAnchors[i][0]
        deviations[i] = np.append((groundTruth[0:2] - anchor[0:2]) / (anchor[2:4]), np.log(groundTruth[2:4] / anchor[2:4]))
    return deviations

# returns a random sample of positive and negative labeled anchors with at most 1:1 ratio positve:negative, but still favoring
# positives, since there will be many, many more negatives to choose from
def getMiniBatch(labels, deviations, batchSize):
    numPositives = min(len(labels[0]), np.random.randint(max(1, int(batchSize * .3)), max(1, int(batchSize * .5))))
    numNegatives = batchSize - numPositives
    positiveIndices = random.sample(range(len(labels[0])), numPositives)
    negativeIndices = random.sample(range(len(labels[1])), numNegatives)
    return ([(labels[0][i][1], labels[0][i][3]) for i in positiveIndices] + [(labels[1][i][1], labels[1][i][2]) for i in negativeIndices], [(labels[0][i][1], deviations[i]) for i in positiveIndices])

def getPixelBox(deviations, anchor, width, height):
    anchor = anchor * np.array([width, height, width, height])
    return np.append(deviations[0:2] * anchor[2:4] + anchor[0:2], anchor[2:4] * np.exp(deviations[2:4]))

# calculates training output for the given imageName and tiling size as a set of nested python
# lists
def getTiledTrainingData(imageName, imageData, tiles):
    nbbs = normalizedBoundingBoxes(imageName, imageData)
    return tiledTrainingData(nbbs, tiles)

# calculates training output for the given bounding box data and tiling size
# as set of nested python lists
def tiledTrainingData(boxes, tiles):
    training = [0] * tiles
    for i in range(0, tiles):
        training[i] = [0] * tiles
        for j in range(0, tiles):
            training[i][j] = [0] * 5
    
    tileDelta = 1 / tiles
    for box in boxes:
        tileX = int(box[KEY_X_CENTER] / tileDelta)
        tileY = int(box[KEY_Y_CENTER] / tileDelta)
        training[tileX][tileY] = [1, box[KEY_X_CENTER], box[KEY_Y_CENTER], box[KEY_WIDTH], box[KEY_HEIGHT]]
    
    return training
    
# calculates the bounding box of a list of points
def boundingBox(points):
    minX = float('inf')
    minY = float('inf')
    maxX = -float('inf')
    maxY = -float('inf')
    
    for p in points:
        minX = min(minX, p[0])
        minY = min(minY, p[1])
        maxX = max(maxX, p[0])
        maxY = max(maxY, p[1])
    
    return {KEY_X_CENTER: (minX + maxX) / 2, KEY_Y_CENTER: (minY + maxY) / 2, KEY_WIDTH: maxX - minX, KEY_HEIGHT: maxY - minY}