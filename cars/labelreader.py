'''
Created on Oct 22, 2018

The loadImageLabelData function takes a file path and creates a dict object
containing a list of the names of the labelled images as well as a name indexed
dict containing the geometric and semantic data concerning the cars in each image

See tools.py for examples of how to use the structure

@author: hauckjp
'''

import json

KEY_ID = 'ID'
KEY_NAME = 'External ID'
KEY_URL = 'Labeled Data'
KEY_LABEL_DATA = 'Label'
KEY_CAR_LIST = 'Vehicle'
KEY_MODEL = 'car_model'
KEY_GEOM = 'geometry'
KEY_COLOR = 'color'

LOCAL_KEY_ID = 'id'
LOCAL_KEY_URL = 'url'
LOCAL_KEY_CARS = 'cars'
LOCAL_KEY_DATA = 'data'
LOCAL_KEY_NAMES = 'names'

def skipCondition(value):
    return value == 'Skip'

def loadImageLabelData(fileName):
    jsonFile = open(fileName)       # Read file containing JSON export data
    jsonString = jsonFile.read()
    jsonFile.close()
    
    jsonData = json.loads(jsonString)   # Use json module to convert the string read from the file into a dict
    
    imageData = {}      # initialize simplified data structures to be returned
    imageNames = []
    
    for car in jsonData:    # JSON object will be a list of dicts, one for each image, so loop over it and 
                            # fill return data structures with important data
        label = car[KEY_LABEL_DATA]     # retrieve the *complete* label data for the current image car
        carData = []                    # initialize simplified data storage
        if not skipCondition(label):       # Leave simplified storage empty if there is no label data (skipCondition = TRUE)
                                        # otherwise extract the car data for the current image
                                        # which may be stored as a list; I don't know what the utility of this is, but I checked,
                                        # and all of the images in the download I did, if they were stored as a list at this point,
                                        # had only one element, so I've just added a check to handle that case in addition to the
                                        # non-list case. If the list utility should ever be used in the future, this part would
                                        # need to be changed.
            if type(label) == list:
                carData = label[0][KEY_CAR_LIST]
            else:
                carData = label[KEY_CAR_LIST]
        
        imageID = car[KEY_ID]           # Extract other necessary data
        imageName = car[KEY_NAME]
        imageURL = car[KEY_URL]
        
        imageNames.append(imageName)    # Store the current image's data in the simplified structures to be returned
        imageData[imageName] = {LOCAL_KEY_ID: imageID, LOCAL_KEY_URL: imageURL, LOCAL_KEY_CARS: carData}
    
    return {LOCAL_KEY_NAMES: imageNames, LOCAL_KEY_DATA: imageData}     
    # return dict with list of image names and a dict mapping those names to
    # their ids, urls, and carData
