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

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
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
    model_path = './model/java-large/java-large-model/model_iter52.release.data-00000-of-00001'

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

    print(model.print_hyperparams())

    #data processing
    print('initialize data')
    collect_java_files('/home/lizhuo/DeepSBST/code2seq/data/example')
    raw_data = extract_method_from_java()
    input = convert_raw_data_to_input(raw_data,config)

    print(input)
    print(filelist)
    '''
    
    
    #refactoring
    for key in input:
        refactored_mn = TestSuiteGenerator.generateTestSuite(model, input[key], raw_data[key], chunk_size, gen, mutation_rate)
        raw_data[key] = refactored_mn


    #retrain and valuate the model
    #os.system('sh train.sh')
    '''

