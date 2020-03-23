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

    def __init__(self,r): 
        self.testObjective = NCTestObjective()
        self.tc_mn = None
        self.tc_api = None
        self.tc_token = None
        self.model = None
        self.coverage = 0.0
        self.record = r
        self.minimal = None

    def layerName(self, model, layer):
        layerNames = [layer.name for layer in model.layers]
        return layerNames[layer]

    def setTestCase(self,testCase, api, token):
        self.tc_api = api
        self.tc_mn = testCase
        self.tc_token = token
            
    def setModel(self,model):
        self.model = model

    def get_activations(self): 
        return get_activations_single_layer(self.model, self.tc_mn, self.tc_api, self.tc_token, layerName(self.model, self.testObjective.layer))
        
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
        self.displayRemainingFeatures()
        
    def displayRemainingFeatures(self):
        #print("%s features to be covered for NC"%(self.originalNumOfFeature))
        print('me')
        #print("including %s."%(str(self.feature)))

    def setLayer(self,layer):
        self.layer = layer 
            
    def setFeature(self,feature):
        self.feature = feature


class CCTestObjectiveEvaluation:

    def __init__(self, r):
        self.testObjective = CCTestObjective()
        self.testCase = None
        self.model = None
        self.hidden = None
        self.threshold = None
        self.minimal = None
        self.coverage = 0.0
        self.record = r

    def setTestCase(self, testCase):
        self.testCase = testCase

    def setModel(self, model):
        self.model = model

    def get_activations(self):
        hidden = self.hidden
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=1)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=1)
        alpha11 = np.insert(np.delete(alpha1, -1, axis=0), 0, 0, axis=0)
        alpha22 = np.insert(np.delete(alpha2, -1, axis=0), 0, 0, axis=0)
        alpha = np.abs(alpha1 - alpha11) + np.abs(alpha2 - alpha22)
        return alpha

    def update_features(self):
        self.minimal = 0
        activation = self.get_activations()
        print("activation")
        print(activation)
        print("threshold")
        print(self.threshold)
        features = (np.argwhere(activation > self.threshold)).tolist()
        print("found %s features for CC." % (len(features)))
        print(self.testObjective.originalfeature)
        # print("including %s."%(str(features)))
        for feature in features:
            if feature in self.testObjective.feature:
                self.minimal = 1
                self.testObjective.feature.remove(feature)
            self.testObjective.feature_count[feature[0]] += 1

        self.coverage = 1 - len(self.testObjective.feature) / self.testObjective.originalNumOfFeature
        self.displayCoverage()
        return self.coverage

    def displayCoverage(self):
        print("cell coverage up to now: %.2f\n" % (self.coverage))

    def evaluate(self):
        if self.testObjective.feature == []:
            return True
        else:
            return False

class CCTestObjective:
    def __init__(self):
        self.layer = None
        self.feature = None
        self.originalNumOfFeature = None
        self.originalfeature = None
        self.feature_count = None

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = len(self.feature)
        self.displayRemainingFeatures()

    def setfeaturecount(self):
        self.feature_count = np.zeros((len(self.feature)))

    def displayRemainingFeatures(self):
        print("%s features to be covered for CC" % (self.originalNumOfFeature))
        # print("including %s."%(str(self.feature)))

    def setLayer(self, layer):
        self.layer = layer

    def setFeature(self, feature):
        self.feature = feature

class SQTestObjectiveEvaluation:

    def __init__(self, r):
        self.testObjective = SQTestObjective()
        self.testCase = None
        self.model = None
        self.hidden = None
        self.symbols = None
        self.minimalp = None
        self.minimaln = None
        self.coverage_p = 0.0
        self.coverage_n = 0.0
        self.record = r

    def setTestCase(self, testCase):
        self.testCase = testCase

    def setModel(self, model):
        self.model = model

    def get_activations(self):
        hidden = self.hidden
        # aggregate positve and negative information
        alpha1 = np.sum(np.where(hidden > 0, hidden, 0), axis=1)
        alpha2 = np.sum(np.where(hidden < 0, hidden, 0), axis=1)
        return alpha1, alpha2

    def update_features(self, indices):
        self.minimalp = 0
        self.minimaln = 0
        activation1, activation2 = self.get_activations()
        print("activation1")
        print(activation1)
        print("activation2")
        print(activation2)
        dat_znorm_p = znorm(activation1[indices])
        dat_znorm_n = znorm(activation2[indices])
        symRep_p = ts_to_string(dat_znorm_p,cuts_for_asize(self.symbols))
        symRep_n = ts_to_string(dat_znorm_n,cuts_for_asize(self.symbols))
        # transfer result to feature(string to tuple)
        feature_p = tuple(symRep_p)
        feature_n = tuple(symRep_n)
        print("found symbolic feature for SQ-P:", symRep_p)
        if feature_p in self.testObjective.feature_p:
            self.minimalp = 1
            self.testObjective.feature_p.remove(feature_p)
        self.coverage_p = 1 - len(self.testObjective.feature_p) / self.testObjective.originalNumOfFeature
        self.displayCoverage1()

        print("found symbolic feature for SQ-N:", symRep_n)
        if feature_n in self.testObjective.feature_n:
            self.minimaln = 1
            self.testObjective.feature_n.remove(feature_n)
        self.coverage_n = 1 - len(self.testObjective.feature_n) / self.testObjective.originalNumOfFeature
        self.displayCoverage2()

        return self.coverage_p, self.coverage_n

    def displayCoverage1(self):
        print("positive sequence coverage up to now: %.2f\n" % (self.coverage_p))

    def displayCoverage2(self):
        print("negative sequence coverage up to now: %.2f" % (self.coverage_n))
        print("-----------------------------------------------------")

    def evaluate(self):
        if self.testObjective.feature == []:
            return True
        else:
            return False

class SQTestObjective:

    def __init__(self):
        self.layer = None
        self.feature_p = None
        self.feature_n = None
        self.originalNumOfFeature = None

    def setOriginalNumOfFeature(self):
        self.originalNumOfFeature = len(self.feature_p)
        self.displayRemainingFeatures()

    def displayRemainingFeatures(self):
        print("%s features to be covered for SQ\n" % (self.originalNumOfFeature))
        print("-----------------------------------------------------")

    def setLayer(self, layer):
        self.layer = layer

    def setFeature(self, feature):
        self.feature = feature

# Memory coverage/ cover the information forget between cells
class MCTestObjectiveEvaluation:

    def __init__(self, r):
        self.testObjective = MCTestObjective()
        self.testCase = None
        self.model = None
        self.hidden = None
        self.threshold = None
        self.minimal = None
        self.coverage = 0.0
        self.record = r

    def setTestCase(self, testCase):
        self.testCase = testCase

    def setModel(self, model):
        self.model = model

    def get_activations(self):
        hidden = self.hidden
        alpha = np.sum(hidden, axis=1) / float(hidden.shape[1])
        # alpha = np.sum(np.where(hidden > 0.8, 1, 0), axis=1)/float(hidden.shape[1])
        return alpha

    def update_features(self):
        self.minimal = 0
        activation = self.get_activations()
        print(activation)
        features = (np.argwhere(activation > self.threshold)).tolist()
        print("found %s features for GC" % (len(features)))
        # print("including %s."%(str(features)))
        for feature in features:
            if feature in self.testObjective.feature:
                self.minimal = 1
                self.testObjective.feature.remove(feature)
            self.testObjective.feature_count[feature[0]] += 1

        self.coverage = 1 - len(self.testObjective.feature) / self.testObjective.originalNumOfFeature
        self.displayCoverage()
        return self.coverage

    def displayCoverage(self):
        print("gate coverage up to now: %.2f\n" % (self.coverage))

    def evaluate(self):
        if self.testObjective.feature == []:
            return True
        else:
            return False

class MCTestObjective:

    def __init__(self):
        self.layer = None
        self.feature = None
        self.originalNumOfFeature = None
        self.originalfeature = None
        self.feature_count = None

    def setOriginalNumOfFeature(self):
        self.originalfeature = self.feature
        self.originalNumOfFeature = len(self.feature)
        self.displayRemainingFeatures()

    def setfeaturecount(self):
        self.feature_count = np.zeros(( len(self.feature)))

    def displayRemainingFeatures(self):
        print("%s features to be covered for GC" % (self.originalNumOfFeature))
        # print("including %s."%(str(self.feature)))

    def setLayer(self, layer):
        self.layer = layer

    def setFeature(self, feature):
        self.feature = feature

