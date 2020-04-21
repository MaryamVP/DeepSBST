import os
import sys
import random
import time
import traceback
from tensorflow.keras.optimizers import RMSprop, Adam
from scipy.stats import rankdata
import math
import numpy as np
from tqdm import tqdm
import argparse
random.seed(42)
import threading 
import configs
import re
import TestSuiteGenerator
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
import SearchEngine
from utils import cos_np, normalize, cos_np_for_normalized, pad, convert, revert
import models, configs, data_loader
import parser
from code_embedding_DeepCS import *
import pandas as pd

    
def parse_args():
    parser = argparse.ArgumentParser("Train and Test Code Search(Embedding) Model")
    parser.add_argument("--data_path", type=str, default='./data/', help="working directory")
    parser.add_argument("--model", type=str, default="JointEmbeddingModel", help="model name")
    parser.add_argument("--dataset", type=str, default="github", help="dataset name")
    parser.add_argument("--mode", choices=["train","eval","repr_code","search"], default='train',
                        help="The mode to run. The `train` mode trains a model;"
                        " the `eval` mode evaluate models in a test set "
                        " The `repr_code/repr_desc` mode computes vectors"
                        " for a code snippet or a natural language description with a trained model.")
    parser.add_argument("--gen", type=int, default='3', help="Number of GA generation")
    parser.add_argument("--chunk_size", type=int, default='20', help="Number of inputs")
    parser.add_argument("--mutation_rate", type=float, default='0.05', help="Mutation Rate")
    parser.add_argument("--verbose",action="store_true", default=True, help="Be verbose")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    config = getattr(configs, 'config_'+args.model)()
    engine = SearchEngine.SearchEngine(args, config)

    ##### Define model ######
    logger.info('Build Model')

    #tf.compat.v1.global_variables_initializer()
    model = getattr(models, args.model)(config)  # initialize the model
    model.build()
    model.summary(export_path = "./output/{}/".format(args.model))
    
    optimizer = config.get('training_params', dict()).get('optimizer', 'adam')
    model.compile(optimizer=optimizer)  

    data_path = args.data_path+args.dataset+'/'


####################################################################################################################################################################
    data_param = config.get('data_params', dict())
    
    ### Loading training set
    data = pd.read_csv('use.rawcode.txt', sep="\n", header=None)
    
    for i in range(len(data[0])):
        method_string = data[0][i]
        code_embedding_DeepCS.replace_methname(method_string, i)
        
    methname = pd.read_csv('new_methname.txt', sep="\n", header=None)

    methnames = code_embedding_DeepCS.text_to_array(methname[0])
    
    
    descs = data_loader.load_hdf5(data_path + data_param['train_desc'], 0, args.chunk_size)
    good_descs = pad(descs, data_param['desc_len'])
    bad_descs = [desc for desc in descs]
    random.shuffle(bad_descs)
    bad_descs = pad(bad_descs, data_param['desc_len'])
    
    
    
    #refactoring
    refactored_mn = TestSuiteGenerator.generateTestSuite(model, methnames, args.chunk_size, args.gen, args.mutation_rate)

    refactored_mn = np.array(refactored_mn).reshape(args.chunk_size, 6)
    methnames = np.concatenate((methnames, refactored_mn), axis=0)

    good_descs = np.concatenate((good_descs, good_descs), axis=0)
    bad_descs = np.concatenate((bad_descs, bad_descs), axis=0)
    
    #retrain the model
    hist = model.fit([methnames, good_descs, bad_descs], epochs=1, batch_size=128, validation_split=0.2)
    
    #evaluate the model
    engine.valid(model, -1 , 10)
    

