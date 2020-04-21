from keract import *
from util import *
from random import *
import numpy as np
from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize
from saxpy.paa import paa
import time

class NCTestObjectiveEvaluation:

    def __init__(self): 
        self.testObjective = NCTestObjective()
        self.tc_input = None
        self.model = None
        self.coverage = 0.0
        self.minimal = None

    def layerName(self, model, layer):
        layerNames = [layer.name for layer in model.layers]
        return layerNames[layer]

    def setTestCase(self,testCase):
        self.tc_input = testCase
            
    def setModel(self,model):
        self.model = model

    def get_activations(self): 
        return get_activations_single_layer(self.model, self.tc_input, layerName(self.model, self.testObjective.layer))
        
    def update_features(self):
        self.minimal = 0
        activation = self.get_activations()
        features = (np.argwhere(activation > 0)).tolist()
        #print("found %s features for NC"%(len(features)))
        for feature in features: 
            if feature in self.testObjective.feature:
                self.minimal = 1
                self.testObjective.feature.remove(feature)

        self.coverage = 1 - len(self.testObjective.feature)/self.testObjective.originalNumOfFeature
        self.displayCoverage()
        return self.coverage
        
    def displayCoverage(self):
        print("neuron coverage up to now: %.2f\n"%(self.coverage))
        pass
                  
    def evaluate(self): 
        if self.testObjective.feature == []: 
            return True
        else: return False

class NCTestObjective:
    def __init__(self):
        self.layer = None
        self.feature = None
        self.originalNumOfFeature = None

    def setOriginalNumOfFeature(self): 
        self.originalNumOfFeature = len(self.feature)

    def setLayer(self,layer):
        self.layer = layer 
            
    def setFeature(self,feature):
        self.feature = feature