import os
import time

class record: 

    def __init__(self,filename,startTime): 

        self.startTime = startTime

        directory = os.path.dirname(filename)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory) 
        self.file = open(filename,"w+") 
        
    def write(self,text): 
        self.file.write(text) 
        
    def close(self): 
        self.file.close()

    def resetTime(self): 
        self.write("reset time at %s\n\n"%(time.time() - self.startTime))
        self.startTime = time.time()

def writeInfo(r,numSamples,numAdv,perturbations,nc_coverage,cc_coverage,mc_coverage,sq_coverage_p,sq_coverage_n):
    r.write("time:%s\n" % (time.time() - r.startTime))
    r.write("samples: %s\n" % (numSamples))
    r.write("neuron coverage: %.3f\n" % (nc_coverage))
    r.write("cell coverage: %.3f\n" % (cc_coverage))
    r.write("gate coverage: %.3f\n" % (mc_coverage))
    r.write("positive sequence coverage: %.3f\n" % (sq_coverage_p))
    r.write("negative sequence coverage: %.3f\n" % (sq_coverage_n))
    r.write("adv. examples: %s\n" % (numAdv))
    r.write("adv. rate: %.3f\n" % (numAdv / numSamples))
    if numAdv > 0 :
        r.write("average perturbation: %.3f\n" % (sum(perturbations) / numAdv))
        r.write("minimum perturbation: %.3f\n\n" % (min(perturbations)))
    else :
        r.write("average perturbation: 0\n")
        r.write("minimum perturbation: 0\n\n")


