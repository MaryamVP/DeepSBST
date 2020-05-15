import os

import random

import argparse
random.seed(42)
from  argparse import *
import TestSuiteGenerator
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
from config import Config
from model import Model
from processing_source_code import *
from interactive_predict import *
import  numpy as np
import tensorflow as tf
filelist = []

def read_file(filename):
    with open(filename, 'r') as file:
        return file.read()


def collect_java_files(filepath):
    dir_list = os.listdir(filepath)
    for dir in dir_list:
        if os.path.isdir(filepath + os.sep + dir):
            collect_java_files(filepath + os.sep + dir)
        elif os.path.isfile(filepath + os.sep + dir):
            if '.java' in filepath + os.sep + dir:
                filelist.append(filepath + os.sep + dir)


def extract_method_from_java():
    raw_data = {}
    for filename in filelist:
        filetext = read_file(filename)
        raw_data[filename] = []
        class_list = extract_class(filetext)[0]
        for class_string in class_list:
            method_list = extract_function(class_string)[0]
            raw_data[filename] = raw_data[filename] + method_list

    return raw_data

def convert_raw_data_to_input(raw_data,config):
    input = {}
    for key in raw_data.keys():
        input[key] = []
        for method in raw_data[key]:
            input[key].append(convert_method_to_input(method,config))
    return input


def save_refactored_data(raw_date, new_raw_data):
    for key in new_raw_data.keys():
        with open(key, 'r') as file:
            filetext = file.read()

        for i in range(len(new_raw_data[key])):
            filetext = filetext.replace(raw_date[i],new_raw_data[i])

        with open(key.replace('.java','_new.java'), 'w') as file:
            file.write(filetext)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False, default= './data')
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False, default= './data')

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False, default= './data')
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False, default= './model/models/java-large-model/model_iter52.release')
    parser.add_argument('--release', action='store_true',
                        help='if specified and loading a trained model, release the loaded model for a smaller model '
                             'size.')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=239)
    return parser.parse_args()

if __name__ == '__main__':
    chunk_size = 20
    gen = 3
    mutation_rate = 0.1
    ##### Define model ######
    logger.info('Build Model')

    #tf.compat.v1.global_variables_initializer()
    print('initialize model')
    args = parse_args()
    np.random.seed(args.seed)

    if args.debug:
        config = Config.get_debug_config(args)
    else:
        config = Config.get_default_config(args)

    model = Model(config)

    #print(model.print_hyperparams())

    #data processing
    print('initialize data')
    data_path  = 'data/example'
    collect_java_files(data_path)
    raw_data = extract_method_from_java()
    input = convert_raw_data_to_input(raw_data,config)

    #every java file as an individual
    print(len(raw_data))
    print(len(input))

    if len(raw_data) != len(input):
        print('the length of raw_data is not equal to input')
        exit()


    new_raw_data = {}

    #refactoring
    print('start to do the refactoring and generate adverisal samples')
    for key in input.keys():
        new_raw_data[key] = []
        refactored_method = TestSuiteGenerator.generateTestSuite(model, input[key], raw_data[key], 10, mutation_rate = 0.1)
        new_raw_data[key].append(refactored_method)

    print(raw_data)
    print(new_raw_data)
    #save the new data
    #save_refactored_data(raw_data,new_raw_data)
