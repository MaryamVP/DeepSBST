import keras
from keras.layers import *
from keras import *
from keras.models import *
from numpy import pad
import copy
import numpy as np
import keras.backend as K
from keras.preprocessing import sequence
import itertools as iter
from util import lp_norm, powerset
from testObjective import *
from oracle import *
from record import writeInfo
import random
from scipy import io
from eda import *
import configs, models
import os
import matplotlib.pyplot as plt


def check_duplicate(new_t2 , t2):
    if np.array_equal(new_t2, t2) :
        return True
    else:
        return False
    
def mutation(t2_mn, t2_apis ,t2_token):
    
    loc_del_mn = random.randint(0, 5)
    loc_del_api = random.randint(0, 29)
    loc_del_token = random.randint(0, 49)
    
    t2_mn[loc_del_mn] = 0
    t2_apis[loc_del_api] = 0
    t2_token[loc_del_token] = 0
    
    return t2_mn , t2_apis, t2_token


def crossover(t1_mn, t1_apis ,t1_token , t2_mn, t2_apis ,t2_token):

    loc_swap_mn = random.randint(0, 5)
    loc_swap_apis = random.randint(0, 29)
    loc_swap_token = random.randint(0, 49)

    #methodname crossover
    new_t1_mn = np.concatenate((t1_mn[:loc_swap_mn], t2_mn[loc_swap_mn:]))
    new_t2_mn = np.concatenate((t2_mn[:loc_swap_mn], t1_mn[loc_swap_mn:]))

    #api 
    new_t1_apis = np.concatenate((t1_apis[:loc_swap_apis] ,t2_apis[loc_swap_apis:]))
    new_t2_apis = np.concatenate((t2_apis[:loc_swap_apis] ,t1_apis[loc_swap_apis:]))


    #token
    
    new_t1_token = np.concatenate((t1_token[:loc_swap_token] ,t2_token[loc_swap_token:]))
    new_t2_token = np.concatenate((t2_token[:loc_swap_token] ,t1_token[loc_swap_token:]))
    
    return new_t1_mn, new_t1_apis ,new_t1_token , new_t2_mn, new_t2_apis ,new_t2_token
                
    

def generateTestSuite(r, threshold_CC, threshold_MC, symbols_SQ, seq, TestCaseNum, minimalTest, TargMetri,
                               CoverageStop, model, methnames, apis, tokens):
    mutated_mn = []
    mutated_api = []
    mutated_token = []
    
    next_gen_mn = []
    next_gen_api = []
    next_gen_token = []
    
    crossover_mn = []
    crossover_api = []
    crossover_token = []
    
    config = configs.config_JointEmbeddingModel()
    r.resetTime()
    layer = 9
    termin = 0
    ncdata = []
    

    tmp_mn = methnames[0].reshape(1, 6)
    tmp_api = apis[0].reshape(1, 30)
    tmp_token = tokens[0].reshape(1, 50)

    nctoe = NCTestObjectiveEvaluation(r)
    nctoe.model = model._code_repr_model
    nctoe.testObjective.layer = layer
    nctoe.setTestCase(tmp_mn, tmp_api, tmp_token)
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc >= np.min(activations_nc))).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()
    
    
    ############################################################################################
    
    generation_number = 5
    total_sample = 5
    current_coverage =[]
    crossover_rate = 0.7
    mutation_rate = 0.05
    number_of_elites = int(10 * total_sample / 100)
    
    current_gen_mn = []
    current_gen_api = []
    current_gen_token = []

    crossover_index = []
    current_coverage = []
    
    mn_index = []
    api_index = []
    token_index = []
    
    changed_mn_index = []
    changed_api_index = []
    changed_token_index = []

    
    current_gen_mn = methnames[:total_sample]
    current_gen_api = apis[:total_sample]
    current_gen_token = tokens[:total_sample]

    
    for i in range(total_sample):
            
        t2_mn = np.array(current_gen_mn[i]).reshape(1, 6) #methnames[i]
        t2_apis = np.array(current_gen_api[i]).reshape(1, 30) #apis[i]
        t2_token = np.array(current_gen_token[i]).reshape(1, 50) #tokens[i]
        
        if not (t2_mn is None):
            
            # update NC coverage≠
            nctoe.setTestCase(t2_mn, t2_apis, t2_token)
            ncdata.append(nctoe.update_features())
            
            print("statistics: \n" , "Current from 2000 is: " , i)
            nctoe.displayCoverage()
                
            current_coverage.append(nctoe.coverage)
        
        
    for j in range(generation_number):
        
        print("*"*70)
        print('Im in this iteration:' , j)
        print("*"*70)

        sorted_coverage = sorted(range(len(current_coverage)), key=lambda k: current_coverage[k], reverse=True)
        print('sorted_coverage',sorted_coverage)

        
        for list_index in sorted_coverage[:number_of_elites]:
            
            next_gen_mn.append(current_gen_mn[list_index])
            next_gen_api.append(current_gen_api[list_index])
            next_gen_token.append(current_gen_token[list_index])
            
        not_changed_mutation = changed_mutation = not_changed_crossover = changed_crossover = not_changed_at_all = 0
        
        for k, list_index in enumerate(sorted_coverage[number_of_elites:]):
            print("list_index",list_index)
            rand = random.random()
            
            if mutation_rate < rand < crossover_rate:
                crossover_mn.append(current_gen_mn[list_index])
                crossover_api.append(current_gen_api[list_index])
                crossover_token.append(current_gen_token[list_index])
                
                crossover_index.append(list_index)
            
            elif rand < mutation_rate:
                
                t2_mn, t2_apis, t2_token = mutation(current_gen_mn[list_index], current_gen_api[list_index], current_gen_token[list_index])
                
                print(type(t2_mn))
                print(t2_mn)
                print(current_gen_mn[list_index])
                
                if  check_duplicate(t2_mn , current_gen_mn[list_index]) or check_duplicate(t2_apis,current_gen_api[list_index]) or \
                    check_duplicate(t2_token, current_gen_token[list_index]):
                    
                    next_gen_mn.append(current_gen_mn[list_index]) 
                    next_gen_api.append(current_gen_api[list_index])
                    next_gen_token.append(current_gen_token[list_index])
                    

                    not_changed_mutation += 1
                    
                else:
                    next_gen_mn.append(t2_mn)
                    next_gen_api.append(t2_apis)
                    next_gen_token.append(t2_token)
                    
                    changed_mn_index.append(list_index)
                    changed_api_index.append(list_index)
                    changed_token_index.append(list_index)
                    
                    changed_mutation += 1

            else:
                next_gen_mn.append(current_gen_mn[list_index]) 
                next_gen_api.append(current_gen_api[list_index])
                next_gen_token.append(current_gen_token[list_index])
                
                
                not_changed_at_all += 1  
                
        
        print('im done with all items in the sorted list')
        
        print("len(next_gen_mn)", len(next_gen_mn))  
        print("len(next_gen_api)", len(next_gen_api))  
        print("len(next_gen_token)", len(next_gen_token))  
        
        print("len crossover_index: ",len(crossover_index))
        if len(crossover_index) != 0:
            
            for i in range(1,len(crossover_mn),2):
                t1_mn, t1_apis ,t1_token , t2_mn, t2_apis ,t2_token = crossover(crossover_mn[i-1], crossover_api[i-1] ,crossover_token[i-1] , 
                                                                                crossover_mn[i], crossover_api[i] ,crossover_token[i])

                if check_duplicate(crossover_mn[i-1] ,t1_mn) or check_duplicate(crossover_mn[i-1] ,t2_mn) or check_duplicate(crossover_mn[i] ,t2_mn) or\
                   check_duplicate(crossover_mn[i] ,t1_mn) or check_duplicate(t1_mn ,t2_mn) :
                    
                    next_gen_mn.append(crossover_mn[i-1])
                    next_gen_mn.append(crossover_mn[i])
                    
                    not_changed_crossover += 1
                    
                else:
                    
                    next_gen_mn.append(t1_mn)
                    next_gen_mn.append(t2_mn)
                    
                    
                    print('Im in mn crossover yes changed and this iteration of its lenhgt:' , i)
                    print("len(next_gen_mn)", len(next_gen_mn))  
                    print("len(next_gen_api)", len(next_gen_api))  
                    print("len(next_gen_token)", len(next_gen_token))  
                    
                    print("Appended mn:" , crossover_index[i-1] , crossover_index[i])
                    changed_mn_index.append(crossover_index[i-1])
                    changed_mn_index.append(crossover_index[i])
                    
                    
                    changed_crossover += 1

                
                if check_duplicate(crossover_api[i-1] ,t1_apis) or check_duplicate(crossover_api[i-1] ,t2_apis) or check_duplicate(crossover_api[i] ,t2_apis) or\
                   check_duplicate(crossover_api[i] ,t1_apis) or check_duplicate(t1_apis ,t2_apis):
                       
                    next_gen_api.append(crossover_api[i-1])
                    next_gen_api.append(crossover_api[i])
                    
                    not_changed_crossover += 1
                    
                else:
                                    
                    next_gen_api.append(t1_apis)
                    next_gen_api.append(t2_apis)
                    
                    changed_api_index.append(crossover_index[i-1])
                    changed_api_index.append(crossover_index[i])
                    
                    
                    
                    changed_crossover += 1

                
                if check_duplicate(crossover_token[i-1] ,t1_token) or check_duplicate(crossover_token[i-1] ,t2_token) or \
                    check_duplicate(crossover_token[i] ,t2_token) or check_duplicate(crossover_token[i] ,t1_token) or check_duplicate(t1_token ,t2_token):
                       
                        
                    next_gen_token.append(crossover_token[i-1])
                    next_gen_token.append(crossover_token[i])
                    
                    not_changed_crossover += 1
                else:
                    next_gen_token.append(t1_token)
                    next_gen_token.append(t2_token)
                    
                    changed_token_index.append(crossover_index[i-1])
                    changed_token_index.append(crossover_index[i])
                    
                    
                    changed_crossover += 1

            if len(crossover_mn) % 2 != 0:
                print("im in odd")
                next_gen_mn.append(crossover_mn[-1])
                next_gen_api.append(crossover_api[-1])
                next_gen_token.append(crossover_token[-1])
        
        temp_coverage = []
        
        min_index = [len(changed_mn_index), len(changed_api_index), len(changed_token_index)].index(min([len(changed_mn_index), len(changed_api_index), len(changed_token_index)]))
        
        if min_index == 0:
            changed = changed_mn_index
        elif min_index == 1:
            changed = changed_api_index
        else:
            changed = changed_token_index

        for l,item in enumerate(changed):
            
            t2_mn = np.array(current_gen_mn[item]).reshape(1, 6) 
            t2_apis = np.array(current_gen_api[item]).reshape(1, 30) 
            t2_token = np.array(current_gen_token[item]).reshape(1, 50)
            
            if not (t2_mn is None):
                
                # update NC coverage
                nctoe.setTestCase(t2_mn, t2_apis, t2_token)
                ncdata.append(nctoe.update_features())
                
                print('current generation is:' , j)
                print("statistics: \n" , "Current from new size(min) is: " , l)
                nctoe.displayCoverage()
                    
                temp_coverage.append(nctoe.coverage)
        
        for i,item in enumerate(changed):
            current_coverage[item] = temp_coverage[i]
               
        
        current_gen_mn = next_gen_mn
        current_gen_api = next_gen_api
        current_gen_token = next_gen_token
        
        changed_mn_index = []
        changed_api_index = []
        changed_token_index = []
        
        crossover_index = []
        crossover_mn = []
        crossover_api = []
        crossover_token = []
        
        
        next_gen_mn = []
        next_gen_api = []
        next_gen_token = []
               
        print("j" , j)
        if j%10 == 0:
            File_object = open(r"current_gen_mn.txt","a")
            File_object.write("Im in {}th:".format(j) + str(current_gen_mn))
            File_object.write('\n')
            
            File_object = open(r"current_gen_api.txt","a")
            File_object.write("Im in {}th:".format(j) + str(current_gen_api))
            File_object.write('\n')
            
            File_object = open(r"current_gen_token.txt","a")
            File_object.write("Im in {}th:".format(j) + str(current_gen_token))
            File_object.write('\n')
            
       
    mutated_mn = current_gen_mn
    mutated_api = current_gen_api
    mutated_token = current_gen_token
    
    
    
    
    print("End of GA")
        
    return mutated_mn, mutated_api, mutated_token
                
    
