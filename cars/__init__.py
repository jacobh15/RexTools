
import cars.labelreader as lr
import cars.tools as tools
import numpy as np

class AnchorBoxFormat:
    
    defaultPrototypes = tools.genPrototypes([np.array([32, 32])], 256, 256)
    defaultPrototypes.append(np.array([0, 0, 48/256, 48/256]))
    defaultForegroundCutoff = .3
    defaultBackgroundCutoff = .7
    defaultTiles = 8
    
    def __init__(self, anchorPrototypes = defaultPrototypes, foregroundCutoff = defaultForegroundCutoff,
                        backgroundCutoff = defaultBackgroundCutoff, tiles = defaultTiles):
        self.prototypes = anchorPrototypes
        self.foregroundCutoff = foregroundCutoff
        self.backgroundCutoff = backgroundCutoff
        self.tiles = tiles

class LabelBoxInterpreter:

    def __init__(self, downloadPath):
        self.imageImport = lr.loadImageLabelData(downloadPath)
        self.names = self.imageImport['names']
        self.data = self.imageImport['data']
    
    def downloadImages(self, images, displayProgress = True):
        imageNames = self.getNames(images)
        
        if displayProgress:
            def display(n):
                tools.printProgressBar(n, len(images), prefix = 'Downloading', suffix = 'of images downloaded', length = 50)
            dl = tools.downloadImages(imageNames, self.data, eachTime = display)
            return dl
        else:
            return tools.downloadImages(imageNames, self.data)
    
    def getNames(self, images):
        imageNames = images
        if type(images[0]) != str:
            imageNames = []
            for i in images:
                imageNames.append(self.names[i])
        return imageNames
    
    def getNormalizedGroundTruthBoxes(self, images):
        imageNames = self.getNames(images)
        
        boxes = [0] *  len(images)
        for i, name in enumerate(imageNames):
            boxes[i] = tools.normalizedBoundingBoxes(name, self.data)
            for j in range(len(boxes[i])):
                boxes[i][j] = tools.boxDictToArray(boxes[i][j])
        return boxes
    
    def getGroundTruthBoxes(self, images):
        imageNames = self.getNames(images)
        
        boxes = [0] * len(images)
        for i, name in enumerate(imageNames):
            boxes[i] = tools.boundingBoxes(name, self.data)
            for j in range(len(boxes[i])):
                boxes[i][j] = tools.boxDictToArray(boxes[i][j])
        return boxes
    
    def createPillowImage(self, image):
        if type(image) == str:
            return tools.getImage(image, self.data)
        else:
            return tools.getImage(self.names[image], self.data)
    
    def createImageWithBoundingBoxes(self, image):
        im = self.createPillowImage(image)
        boxes = self.getGroundTruthBoxes([image])[0]
        for box in boxes:
            tools.drawBoundingBox(im, box, normalized = False)
        return im
    
    def getTensorFlowImages(self, images, displayProgress = True):
        imageNames = self.getNames(images)
        
        if displayProgress:
            def display(n):
                tools.printProgressBar(n, len(images), prefix = "Downloading", suffix = "images downloaded", length = 50)
            dl = tools.getTensorFlowImages(imageNames, self.data, eachTime = display)
            return dl
        else:
            return tools.getTensorFlowImages(imageNames, self.data)
    
    def getPixelBoxes(self, deviations, images, anchors):
        names = self.getNames(images)
        pixelBoxes = []
        for i in range(len(names)):
            w = self.data[names[i]]['image']['width']
            h = self.data[names[i]]['image']['height']
            thisImage = []
            for j in range(len(deviations[i])):
                thisImage.append(tools.getPixelBox(deviations[i][j], anchors[i][j], w, h))
            pixelBoxes.append(thisImage)
        return pixelBoxes
    
    def getMiniBatches(self, images, anchors, batchSize):
        labels, deviations = self.getAnchorTrainingData(images, anchors)
        batches = []
        for i in range(0, len(images)):
            batches.append(tools.getMiniBatch(labels[i], deviations[i], batchSize))
        return batches
    
    def getAnchorTrainingData(self, images, anchors):
        labels = []
        deviations = []
        names = self.getNames(images)
        boxes = self.getNormalizedGroundTruthBoxes(images)
        for i in range(len(names)):
            w = self.data[names[i]]['image']['width']
            h = self.data[names[i]]['image']['height']
            label = tools.getLabeledAnchors(boxes[i], anchors.tiles, anchors.prototypes, anchors.foregroundCutoff, anchors.backgroundCutoff, w, h)
            labels.append(label)
            deviation = tools.getAnchorDeviations(boxes[i], label[0], w, h)
            deviations.append(deviation)
        return (labels, deviations)
    