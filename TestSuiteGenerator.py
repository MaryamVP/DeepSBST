from numpy import pad
import numpy as np
from testObjective import *
import random


def set_neuron_coverage(model , model_input):
    layer = 1

    nctoe = NCTestObjectiveEvaluation()
    #load_model here
    nctoe.model = model._desc_repr_model
    nctoe.testObjective.layer = layer
    nctoe.setTestCase(model_input)
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc >= np.min(activations_nc))).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()
    
    return nctoe

def calculate_neuron_coverage(model_input , nctoe):
    
    nc_data = []
    current_coverage = []

    if model_input is None:
        print('The input is empty, check the data and try again!')
        
    else:
        # update NC coverage
        nctoe.setTestCase(model_input)
        nc_data.append(nctoe.update_features())
        nctoe.displayCoverage()
            
        current_coverage.append(nctoe.coverage)

    return current_coverage



def log_tracking(current_iteration, current_gen):
    if current_iteration%10 == 9:
                File_object = open(r"current_gen.txt","a")
                File_object.write("Current Iteration is: {}th:".format(current_iteration) + str(current_gen))
                File_object.write('\n')
        
def check_duplicate(new_t2 , t2):
    if np.array_equal(new_t2, t2) :
        return True
    else:
        return False
    
def mutation(a):
    
    loc_del = random.randint(0, len(a))
    a[loc_del] = 0
    
    return a


def crossover(a, b):

    loc_swap = random.randint(0, min(len(a), len(b)))

    new_a = np.concatenate((a[:loc_swap], b[loc_swap:]))
    new_b = np.concatenate((b[:loc_swap], a[loc_swap:]))

    return new_a, new_b
                      

    
def GA(generation_number, current_coverage, number_of_elites, mutation_rate, crossover_rate):
        
    current_gen = []
    next_gen = []
    crossover = []
    crossover_index = []
    changed_index = []
    temp_coverage = []
    
    for current_iteration in range(generation_number):
        
        sorted_coverage = sorted(range(len(current_coverage)), key=lambda k: current_coverage[k], reverse=True)

        for list_index in sorted_coverage[:number_of_elites]:
            
            next_gen.append(current_gen[list_index])
        
        for k, list_index in enumerate(sorted_coverage[number_of_elites:]):
            rand = random.random()
            
            if mutation_rate < rand < crossover_rate:
                crossover.append(current_gen[list_index])
                crossover_index.append(list_index)
            
            elif rand < mutation_rate:
                mutated_input = mutation(current_gen[list_index])
                
                if  check_duplicate(mutated_input , current_gen[list_index]):
                    next_gen.append(current_gen[list_index])
                    
                else:
                    next_gen.append(mutated_input)
                    changed_index.append(list_index)

            else:
                next_gen.append(current_gen[list_index])
                
                
        if len(crossover_index) != 0:
            
            for i in range(1,len(crossover),2):
                
                crossovered_input_1, crossovered_input_2 = crossover(crossover[i-1], crossover[i])

                if check_duplicate(crossover[i-1] ,crossovered_input_1) or check_duplicate(crossover[i-1] ,crossovered_input_2) or check_duplicate(crossover[i] ,crossovered_input_2) or\
                check_duplicate(crossover[i] ,crossovered_input_1) or check_duplicate(crossovered_input_1 ,crossovered_input_2) :
                    
                    next_gen.append(crossover[i-1])
                    next_gen.append(crossover[i])

                else:
                    
                    next_gen.append(crossovered_input_1)
                    next_gen.append(crossovered_input_2)
                    
                    changed_index.append(crossover_index[i-1])
                    changed_index.append(crossover_index[i])


            if len(crossover) % 2 != 0:
                print("Odd length for crossover")
                next_gen.append(crossover[-1])

        
        for i,item in enumerate(changed_index):
            print('Current iteration in the changed_index:' , i)
            current_coverage[item] = calculate_neuron_coverage(current_gen[item] , nctoe)

        current_gen = next_gen

        changed_index = []
        crossover_index = []
        crossover = []
        next_gen= [] 

        log_tracking(current_iteration, current_gen)
        
        
    mutated_input = current_gen
    
    return mutated_input

def generateTestSuite(model, model_input):
    
    generation_number = 3
    total_sample = 10
    crossover_rate = 0.7
    mutation_rate = 0.05
    number_of_elites = int(total_sample / 10)
    
    current_coverage = []
    
    nctoe = set_neuron_coverage(model, model_input)

    for i in range(total_sample):
        
        print("Current iteration is: " , i)
        current_coverage.append(calculate_neuron_coverage(np.array(model_input[i]), nctoe))
                
    print('Current Coverage is:', current_coverage)
    
    mutated_input = GA(generation_number, current_coverage, number_of_elites, mutation_rate, crossover_rate)
    
    return mutated_input