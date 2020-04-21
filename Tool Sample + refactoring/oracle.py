import numpy as np
import os
import time
from util import lp_norm

class oracle: 

    def __init__(self,input,lp,radius): 

        self.input = input
        self.measurement = lp
        self.radius = radius
        
    def passOracle(self,test):
        n = np.count_nonzero(self.input- test)
        return np.linalg.norm(self.input - test,ord=self.measurement)/float(n) <= self.radius

    def measure(self,test): 
        return np.linalg.norm(self.input - test,ord=self.measurement)