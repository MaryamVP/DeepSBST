from numpy import pad
import numpy as np
from testObjective import *
import random
import GA_refactoring
from interactive_predict import *

def set_neuron_coverage(model, model_input):
    layer = 1

    nctoe = NCTestObjectiveEvaluation()
    # load_model here
    nctoe.model = model._code_repr_model
    nctoe.testObjective.layer = layer
    nctoe.setTestCase(model_input)
    activations_nc = nctoe.get_activations()
    nctoe.testObjective.feature = (np.argwhere(activations_nc)).tolist()
    nctoe.testObjective.setOriginalNumOfFeature()

    return nctoe


def calculate_neuron_coverage(model_input, nctoe):
    nc_data = []
    current_coverage = []

    if model_input is None:
        print('The input is empty, check the data and try again!')

    else:
        # update NC coverage
        nctoe.setTestCase(model_input)
        nc_data.append(nctoe.update_features())
        # nctoe.displayCoverage()

        current_coverage.append(nctoe.coverage)

    return current_coverage


def log_tracking(current_iteration, current_gen):
    if current_iteration % 10 == 9:
        File_object = open(r"current_gen.txt", "a")
        File_object.write("Current Iteration is: {}th:".format(current_iteration) + str(current_gen))
        File_object.write('\n')


def check_duplicate(new_t2, t2):
    if np.array_equal(new_t2, t2):
        return True


def mutation(method):

    mutation_method = random.choice(
                                    GA_refactoring.rename_argument,
                                    GA_refactoring.return_optimal,
                                    GA_refactoring.add_argumemts,
                                    GA_refactoring.enhance_for_loop,
                                    GA_refactoring.enhance_filed,
                                    GA_refactoring.enhance_if,
                                    GA_refactoring.rename_api,
                                    GA_refactoring.rename_local_variable,
                                    GA_refactoring.add_local_variable,
                                    GA_refactoring.rename_method_name,
                                    GA_refactoring.add_print
                                    )

    new_method = mutation_method(method_string)
    return new_method


def GA(generation_number, current_gen, current_raw, current_coverage, number_of_elites, mutation_rate, nctoe):
    next_gen = []
    next_raw = []
    changed_index = []
    temp_coverage = []

    for current_iteration in range(generation_number):

        sorted_coverage = sorted(range(len(current_coverage)), key=lambda k: current_coverage[k], reverse=True)

        for list_index in sorted_coverage[:number_of_elites]:
            next_gen.append(current_gen[list_index])
            next_raw.append(current_raw[list_index])

        for k, list_index in enumerate(sorted_coverage[number_of_elites:]):
            rand = random.random()

            if rand < mutation_rate:

                mutated_input = mutation(current_raw[list_index])
                current_gen[list_index] = convert_method_to_input(mutated_input)

                if check_duplicate(mutated_input, current_gen[list_index]):
                    next_gen.append(current_gen[list_index])
                    next_raw.append(current_raw[list_index])

                else:
                    next_gen.append(current_gen[list_index])
                    next_raw.append(current_raw[list_index])
                    changed_index.append(list_index)

            else:
                next_gen.append(current_gen[list_index])

        for i, item in enumerate(changed_index):
            print('Current iteration in the changed_index:', i)
            current_coverage[item] = calculate_neuron_coverage(current_gen[item], nctoe)

        current_gen = next_gen
        current_raw = next_raw
        changed_index = []
        next_gen = []

        log_tracking(current_iteration, current_gen)

    mutated_input = current_raw

    return mutated_input


def generateTestSuite(model, model_input, raw_input, generation_number, mutation_rate):
    number_of_elites = max(1,len(model_input) / 2)

    test_input = []
    current_coverage = []

    for i in range(len(model_input)):
        test_input.append(np.array(model_input[i]).reshape(1, 6))

    nctoe = set_neuron_coverage(model, test_input)

    for i in range(len(model_input)):
        print("Current iteration is: ", i)
        current_coverage.append(calculate_neuron_coverage(test_input[i], nctoe))

    print('Current Coverage is:', current_coverage)

    mutated_raw_input = GA(generation_number, test_input, raw_input, current_coverage, number_of_elites, mutation_rate, nctoe)

    print('mutated_input', mutated_raw_input)

    return mutated_raw_input
